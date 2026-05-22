# ==============================================================================
# app/pipeline/extractor.py
#
# PIPELINE STAGE 1 — Extract
#
# Splits an uploaded video into:
#   • Individual PNG frames  →  storage/frames/{job_id}/frame_NNNN.png
#   • Isolated audio track   →  storage/frames/{job_id}/audio.aac
#
# Both outputs are consumed by later stages:
#   frames/   → batcher.py   → AI models (Tool 1 or 2)
#   audio.aac → recomposer.py → muxed back onto the final output video
#
# Design decisions:
#   • PNG is lossless — critical because latent diffusion models are
#     sensitive to JPEG compression artefacts in input frames.
#   • Audio is always transcoded to AAC 192 kbps for maximum compatibility
#     (.mov PCM, .webm Opus → AAC).  This allows recomposer.py to use
#     -c:a copy (no re-encode) and avoids codec mismatch errors.
#   • All FFmpeg calls run inside asyncio.to_thread() so they never block
#     the event loop while the frontend SSE connection is held open.
#   • Frame filenames use a zero-padded 6-digit counter (frame_000001.png)
#     to support up to 999,999 frames — ~9 hours at 30 fps.
# ==============================================================================

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Awaitable, Callable, Optional

import cv2

from app.core.config import settings

logger = logging.getLogger(__name__)

# Type alias for the async progress callback used throughout the pipeline.
# Signature: callback(pct: int, message: str, detail: str) -> None
ProgressCB = Callable[[int, str, str], Awaitable[None]]


# ── Public API ─────────────────────────────────────────────────────────────────

async def extract(
    upload_path: Path,
    job_id:      str,
    progress_cb: Optional[ProgressCB] = None,
) -> tuple[Path, Optional[Path], dict]:
    """
    Run Stage 1 of the pipeline: extract frames and audio from the uploaded video.

    Args:
        upload_path:  Path to the uploaded file (e.g. storage/uploads/{job_id}/original.mp4)
        job_id:       8-char hex job identifier
        progress_cb:  Optional async callback(pct, message, detail) for SSE progress events

    Returns:
        frames_dir:   Path to directory containing frame_NNNN.png files
        audio_path:   Path to extracted audio.aac, or None if video has no audio track
        metadata:     dict — fps, width, height, frame_count, duration_s

    Raises:
        FileNotFoundError:  upload_path does not exist
        RuntimeError:       FFmpeg subprocess exits with non-zero code
    """
    if not upload_path.exists():
        raise FileNotFoundError(f"Upload file not found: {upload_path}")

    frames_dir = Path(settings.frames_dir) / job_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── A: Read metadata (OpenCV, no decoding) ───────────────────────────────
    await _cb(progress_cb, 2, "Reading video metadata…", upload_path.name)
    metadata = _read_metadata(upload_path)
    logger.info(
        "Job %s  |  %dx%d  %.2f fps  %.1f s  %d frames",
        job_id, metadata["width"], metadata["height"],
        metadata["fps"], metadata["duration_s"], metadata["frame_count"],
    )

    # ── B: Extract every frame as a lossless PNG ─────────────────────────────
    await _cb(progress_cb, 4, "Extracting frames…",
              f"{metadata['frame_count']} frames @ {metadata['fps']:.2f} fps")

    frame_pattern = str(frames_dir / "frame_%06d.png")
    await asyncio.to_thread(_ffmpeg_extract_frames, upload_path, frame_pattern)

    actual_frames = sorted(frames_dir.glob("frame_*.png"))
    if not actual_frames:
        raise RuntimeError(f"FFmpeg extracted 0 frames from {upload_path}. "
                           "Check that ffmpeg is installed and on PATH.")

    await _cb(progress_cb, 8,
              f"Extracted {len(actual_frames)} frames",
              str(frames_dir))
    logger.info("Job %s: %d frames written to %s", job_id, len(actual_frames), frames_dir)

    # ── C: Extract audio track ───────────────────────────────────────────────
    audio_path: Optional[Path] = None
    if _has_audio(upload_path):
        await _cb(progress_cb, 9, "Extracting audio track…", "Transcoding to AAC 192 kbps")
        audio_dest = frames_dir / "audio.aac"
        try:
            await asyncio.to_thread(_ffmpeg_extract_audio, upload_path, audio_dest)
            audio_path = audio_dest
            logger.info("Job %s: audio → %s", job_id, audio_dest)
        except RuntimeError as exc:
            # Non-fatal: recompose will produce a silent video.
            logger.warning("Job %s: audio extraction failed — continuing silently. %s", job_id, exc)
    else:
        logger.info("Job %s: no audio track found, skipping audio extraction", job_id)

    await _cb(progress_cb, 10,
              "Extraction complete",
              f"{len(actual_frames)} frames  |  audio: {'yes' if audio_path else 'none'}")

    return frames_dir, audio_path, metadata


# ── FFmpeg helpers (blocking — called via asyncio.to_thread) ───────────────────

def _ffmpeg_extract_frames(upload_path: Path, frame_pattern: str) -> None:
    """
    Extract every frame of the video as a lossless PNG.

    Equivalent shell command:
        ffmpeg -y -i input.mp4 -vsync vfr -q:v 1 frames/frame_%06d.png

    -vsync vfr : variable frame rate mode — preserves exact timestamps and
                 avoids duplicate frames in VFR source material (common in
                 phone recordings and screen captures).
    """
    cmd = [
        "ffmpeg", "-y",
        "-i",        str(upload_path),
        "-vsync",    "vfr",
        "-q:v",      "1",
        "-hide_banner",
        "-loglevel", "error",
        frame_pattern,
    ]
    _run(cmd, context="frame extraction")


def _ffmpeg_extract_audio(upload_path: Path, audio_dest: Path) -> None:
    """
    Extract the audio track, transcoding to AAC 192 kbps at 44.1 kHz.

    We always transcode (never copy) so that:
      • .webm Opus and .mov ALAC/PCM are reliably converted
      • recomposer.py can use -c:a copy without codec mismatch errors

    Equivalent shell command:
        ffmpeg -y -i input.mp4 -vn -acodec aac -b:a 192k -ar 44100 audio.aac
    """
    cmd = [
        "ffmpeg", "-y",
        "-i",       str(upload_path),
        "-vn",                         # strip video — audio-only pass
        "-acodec",  "aac",
        "-b:a",     "192k",
        "-ar",      "44100",           # normalise sample rate
        "-hide_banner",
        "-loglevel", "error",
        str(audio_dest),
    ]
    _run(cmd, context="audio extraction")


# ── Metadata helpers ───────────────────────────────────────────────────────────

def _read_metadata(upload_path: Path) -> dict:
    """
    Read video properties via OpenCV (header-only, no frame decoding).
    Falls back to sane defaults if the cap fails to open.
    """
    cap = cv2.VideoCapture(str(upload_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV could not open {upload_path}. "
                           "Ensure the file is a valid video and opencv-python is installed.")

    fps         = cap.get(cv2.CAP_PROP_FPS)         or 30.0
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1920
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return {
        "fps":         round(float(fps), 3),
        "width":       width,
        "height":      height,
        "frame_count": frame_count,
        "duration_s":  round(frame_count / fps, 3) if fps > 0 else 0.0,
    }


def _has_audio(upload_path: Path) -> bool:
    """
    Return True if ffprobe detects at least one audio stream.

    Equivalent shell command:
        ffprobe -v error -select_streams a:0 \
                -show_entries stream=codec_type -of csv=p=0 input.mp4
    """
    cmd = [
        "ffprobe",
        "-v",              "error",
        "-select_streams", "a:0",
        "-show_entries",   "stream=codec_type",
        "-of",             "csv=p=0",
        str(upload_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return "audio" in result.stdout
    except Exception as exc:
        logger.debug("ffprobe audio detection failed: %s", exc)
        return False


# ── Subprocess runner ──────────────────────────────────────────────────────────

def _run(cmd: list[str], context: str = "") -> None:
    """
    Run a subprocess synchronously.
    Raises RuntimeError with stderr on non-zero exit — the orchestrator
    surfaces this as an SSE error event to the frontend.
    """
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
    """Fire the progress callback if one is registered."""
    if cb is not None:
        await cb(pct, message, detail)
