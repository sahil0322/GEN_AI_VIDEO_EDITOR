# ==============================================================================
# app/pipeline/recomposer.py
#
# PIPELINE STAGE 4 — Recompose
#
# Stitches all AI-processed frames back into a video and muxes the original
# audio track back on top.
#
# Why FFmpeg and not OpenCV VideoWriter?
#   OpenCV's VideoWriter cannot mux audio.  FFmpeg handles frame-accurate
#   A/V sync, preserves the original audio (no re-encode after our AAC
#   transcode in extractor.py), and produces a universally compatible MP4
#   (H.264 video + AAC audio) in a single subprocess call.
#
# Output quality:
#   H.264 at CRF 18 is near-lossless for most content — the human visual
#   system cannot distinguish CRF 18 from lossless at normal viewing distances.
#   CRF 18 at 1080p produces files of roughly 4–8 MB/min, which is reasonable
#   for a web download.  Adjust via OUTPUT_CRF in .env (Step 10).
#
# Temporal consistency note:
#   This stage has no AI model involvement.  Frame order comes entirely from
#   the sorted glob of processed/*.png files, which are named to match the
#   original frame numbering from extractor.py.  Maintaining that naming
#   convention throughout the pipeline is what preserves frame-level temporal
#   order without any additional synchronisation logic here.
# ==============================================================================

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Awaitable, Callable, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

ProgressCB = Callable[[int, str, str], Awaitable[None]]

# H.264 quality: 0 (lossless) → 51 (worst).  18 is visually near-lossless.
_DEFAULT_CRF = 18
# x264 encoding speed preset.  "medium" balances speed and file size.
_DEFAULT_PRESET = "medium"


# ── Public API ─────────────────────────────────────────────────────────────────

async def recompose(
    processed_dir: Path,
    audio_path:    Optional[Path],
    output_path:   Path,
    fps:           float,
    crf:           int = _DEFAULT_CRF,
    preset:        str = _DEFAULT_PRESET,
    progress_cb:   Optional[ProgressCB] = None,
) -> Path:
    """
    Stage 4: stitch processed frames into a video and mux the original audio.

    Args:
        processed_dir: Directory containing AI-processed frame_NNNN.png files
        audio_path:    Path to audio.aac from extractor.py (None → silent output)
        output_path:   Destination for the final MP4 (created by this function)
        fps:           Original video frame rate (from extractor metadata)
        crf:           H.264 Constant Rate Factor (18 = near-lossless)
        preset:        x264 speed/quality preset (medium recommended)
        progress_cb:   Optional async progress callback

    Returns:
        output_path — the Path of the finished MP4 file

    Raises:
        ValueError:  No processed frames found in processed_dir
        RuntimeError: FFmpeg exits with non-zero code
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Collect and validate processed frames ─────────────────────────────────
    await _cb(progress_cb, 90, "Collecting processed frames…", str(processed_dir))

    frames = _collect_frames(processed_dir)
    if not frames:
        raise ValueError(
            f"No processed frames found in {processed_dir}. "
            "Did the inference stage complete successfully?"
        )

    logger.info(
        "Recomposing job output: %d frames @ %.2f fps  →  %s",
        len(frames), fps, output_path,
    )

    # ── Write a numbered symlink tree so FFmpeg can use %06d pattern ──────────
    # Processed frames may have non-contiguous numbers if overlap de-duplication
    # removed some frames.  We create a temp directory with sequentially
    # renamed symlinks so FFmpeg's -i frame_%06d.png pattern works correctly.
    await _cb(progress_cb, 91, "Preparing frame sequence…",
              f"{len(frames)} frames to stitch")

    with tempfile.TemporaryDirectory(prefix="flowedit_recompose_") as tmpdir:
        tmp_path = Path(tmpdir)
        _create_sequential_symlinks(frames, tmp_path)

        frame_pattern = str(tmp_path / "frame_%06d.png")

        # ── FFmpeg: stitch frames ──────────────────────────────────────────────
        await _cb(progress_cb, 93, "Stitching frames with FFmpeg…",
                  f"H.264 CRF {crf}, preset {preset}")

        if audio_path and audio_path.exists():
            # One-pass: encode video + copy audio simultaneously
            await asyncio.to_thread(
                _ffmpeg_stitch_with_audio,
                frame_pattern, audio_path, output_path, fps, crf, preset,
            )
        else:
            # Silent video: no audio input
            if audio_path:
                logger.warning("audio_path %s does not exist — producing silent output", audio_path)
            await asyncio.to_thread(
                _ffmpeg_stitch_silent,
                frame_pattern, output_path, fps, crf, preset,
            )

    # Verify the output was actually created
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg produced no output at {output_path}")

    size_mb = output_path.stat().st_size / 1_048_576
    await _cb(progress_cb, 99,
              "Recomposition complete",
              f"{output_path.name}  ({size_mb:.1f} MB)")

    logger.info("Recomposition done: %s  (%.1f MB)", output_path, size_mb)
    return output_path


# ── FFmpeg helpers ─────────────────────────────────────────────────────────────

def _ffmpeg_stitch_with_audio(
    frame_pattern: str,
    audio_path:    Path,
    output_path:   Path,
    fps:           float,
    crf:           int,
    preset:        str,
) -> None:
    """
    Stitch frames + mux original audio in a single FFmpeg pass.

    Equivalent shell command:
        ffmpeg -y \
          -r 30 -i frames/frame_%06d.png \
          -i audio.aac \
          -c:v libx264 -pix_fmt yuv420p -crf 18 -preset medium \
          -c:a copy \
          -shortest \
          output.mp4

    Flags explained:
      -r fps          Input frame rate (must match source FPS exactly so A/V sync is correct)
      -pix_fmt yuv420p  Required for maximum browser/player compatibility (QuickTime, VLC, web)
      -c:a copy         Copy AAC stream without re-encoding — preserves quality and is instant
      -shortest         Truncate output to the shorter of video/audio streams (handles rounding)
    """
    cmd = [
        "ffmpeg", "-y",
        "-r",          str(fps),
        "-i",          frame_pattern,
        "-i",          str(audio_path),
        "-c:v",        "libx264",
        "-pix_fmt",    "yuv420p",
        "-crf",        str(crf),
        "-preset",     preset,
        "-c:a",        "copy",
        "-shortest",
        "-movflags",   "+faststart",   # move metadata to file start for streaming
        "-hide_banner",
        "-loglevel",   "error",
        str(output_path),
    ]
    _run(cmd, context="stitch+audio mux")


def _ffmpeg_stitch_silent(
    frame_pattern: str,
    output_path:   Path,
    fps:           float,
    crf:           int,
    preset:        str,
) -> None:
    """
    Stitch frames into a silent MP4 (no audio track).

    Equivalent shell command:
        ffmpeg -y -r 30 -i frames/frame_%06d.png \
               -c:v libx264 -pix_fmt yuv420p -crf 18 -preset medium output.mp4
    """
    cmd = [
        "ffmpeg", "-y",
        "-r",        str(fps),
        "-i",        frame_pattern,
        "-c:v",      "libx264",
        "-pix_fmt",  "yuv420p",
        "-crf",      str(crf),
        "-preset",   preset,
        "-movflags", "+faststart",
        "-hide_banner",
        "-loglevel", "error",
        str(output_path),
    ]
    _run(cmd, context="silent stitch")


# ── Frame collection and renaming ──────────────────────────────────────────────

def _collect_frames(processed_dir: Path) -> list[Path]:
    """
    Return all frame_*.png files in processed_dir, sorted by frame number.
    Uses the same numeric sort as batcher._collect_frames for consistency.
    """
    frames = list(processed_dir.glob("frame_*.png"))
    frames.sort(key=lambda p: _frame_number(p))
    return frames


def _frame_number(p: Path) -> int:
    """Extract integer frame number from frame_000042.png → 42."""
    try:
        return int(p.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _create_sequential_symlinks(frames: list[Path], tmp_dir: Path) -> None:
    """
    Create sequentially-numbered symlinks in tmp_dir so FFmpeg's -i
    frame_%06d.png pattern works even if the source frame numbers have gaps
    (which happens when batcher overlap de-duplication removes some frames).

    frame_000003.png → tmp/frame_000001.png
    frame_000004.png → tmp/frame_000002.png
    ...

    We use symlinks (not copies) to avoid duplicating potentially large PNG files.
    On Windows where symlinks require elevated privileges, fall back to hard links.
    """
    for idx, src in enumerate(frames, start=1):
        dst = tmp_dir / f"frame_{idx:06d}.png"
        try:
            dst.symlink_to(src.resolve())
        except (OSError, NotImplementedError):
            # Windows fallback — hard links don't require elevated privileges
            try:
                dst.hardlink_to(src.resolve())
            except Exception:
                # Last resort: copy the file (slow but always works)
                import shutil
                shutil.copy2(src, dst)


# ── Subprocess runner ──────────────────────────────────────────────────────────

def _run(cmd: list[str], context: str = "") -> None:
    logger.debug("FFmpeg [%s]: %s", context, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg {context} failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )


# ── Progress callback helper ───────────────────────────────────────────────────

async def _cb(
    cb:      Optional[ProgressCB],
    pct:     int,
    message: str,
    detail:  str = "",
) -> None:
    if cb is not None:
        await cb(pct, message, detail)
