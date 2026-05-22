# ==============================================================================
# app/pipeline/batcher.py
#
# PIPELINE STAGE 2 — Batch
#
# Groups a sorted list of frame paths into fixed-size chunks so Stage 3
# can process them one batch at a time without loading all frames into VRAM
# at once.
#
# Why this is its own module (not inside the model code):
#   Batch size is a hardware constraint (VRAM), not a model concern.
#   Keeping it here means you can tune INFER_BATCH_SIZE in .env once and
#   every model — ProPainter, E2FGVI, SVD, ControlNet — automatically
#   gets the right chunk size without any model code changes.
#
# Temporal consistency note:
#   Some models (ProPainter, RAFT) need a small overlap between consecutive
#   batches so they can compute inter-batch optical flow without discontinuities
#   at batch boundaries.  The `overlap` parameter implements this:
#
#     batch 0:  frames [0 … 7]
#     batch 1:  frames [6 … 13]   ← frames 6-7 overlap with batch 0
#     batch 2:  frames [12 … 19]  ← frames 12-13 overlap with batch 1
#
#   The orchestrator strips the overlapping frames from each batch's output
#   before writing to processed/ so they don't appear twice in the final video.
#   Set overlap=0 to disable (fine for style_transfer where RAFT handles
#   inter-frame consistency at the pixel level, not the batch level).
# ==============================================================================

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

def make_batches(
    frames_dir:  Path,
    batch_size:  int | None = None,
    overlap:     int = 2,
) -> list[list[Path]]:
    """
    Collect and sort all extracted frames, then split them into batches.

    Args:
        frames_dir:  Directory produced by extractor.py containing frame_NNNN.png files
        batch_size:  Frames per batch (defaults to settings.infer_batch_size)
        overlap:     Number of frames each batch shares with the next batch.
                     Prevents hard-cut artefacts at batch boundaries for models
                     that use temporal context (ProPainter, RAFT).
                     Set to 0 for style_transfer (RAFT handles this internally).

    Returns:
        List of batches, where each batch is a sorted list of Path objects.
        The last batch may be smaller than batch_size.

    Raises:
        FileNotFoundError: frames_dir does not exist
        ValueError:        no frames found in frames_dir
    """
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    batch_size = batch_size or settings.infer_batch_size
    if batch_size < 1:
        raise ValueError(f"batch_size must be ≥ 1, got {batch_size}")
    if overlap < 0 or overlap >= batch_size:
        raise ValueError(f"overlap must be in [0, batch_size), got overlap={overlap} batch_size={batch_size}")

    frames = _collect_frames(frames_dir)
    if not frames:
        raise ValueError(f"No frame_*.png files found in {frames_dir}. "
                         "Did extractor.py run successfully?")

    batches = list(_chunk(frames, batch_size=batch_size, overlap=overlap))

    logger.info(
        "Batched %d frames → %d batches  (size=%d overlap=%d)",
        len(frames), len(batches), batch_size, overlap,
    )
    return batches


def total_output_frames(batches: list[list[Path]], overlap: int) -> int:
    """
    Return the number of unique output frames the orchestrator should expect
    after de-overlapping.  Used to compute accurate inference progress %.

    With overlap=0 this equals sum(len(b) for b in batches).
    With overlap>0 the overlapping frames from each batch are discarded,
    so the total is slightly less.
    """
    if not batches:
        return 0
    if overlap == 0:
        return sum(len(b) for b in batches)
    # First batch contributes all frames; subsequent batches contribute (size - overlap)
    first = len(batches[0])
    rest  = sum(max(0, len(b) - overlap) for b in batches[1:])
    return first + rest


def output_slice(batch_idx: int, batch: list[Path], overlap: int) -> list[Path]:
    """
    Return the slice of a batch whose output frames should be written to
    processed/.  Discards the leading `overlap` frames from every batch
    except the first (those frames were already output by the previous batch).

    Args:
        batch_idx: 0-based index of this batch
        batch:     The full list of Path objects in the batch
        overlap:   Same overlap value passed to make_batches()

    Returns:
        Subset of `batch` whose processed outputs should be kept.
    """
    if batch_idx == 0 or overlap == 0:
        return batch
    return batch[overlap:]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _collect_frames(frames_dir: Path) -> list[Path]:
    """
    Return all frame_*.png files in frames_dir, sorted by frame number.

    Sorting is numeric (frame_000001 before frame_000010) not lexicographic,
    so the order is correct even if frame counts exceed single digits.
    """
    frames = list(frames_dir.glob("frame_*.png"))
    frames.sort(key=_frame_number)
    return frames


def _frame_number(p: Path) -> int:
    """
    Extract the integer frame number from a path like frame_000042.png.
    Falls back to 0 so malformed filenames sort to the front rather than crash.
    """
    stem = p.stem  # "frame_000042"
    try:
        return int(stem.split("_", 1)[1])
    except (IndexError, ValueError):
        logger.warning("Could not parse frame number from %s", p.name)
        return 0


def _chunk(
    frames:     list[Path],
    batch_size: int,
    overlap:    int,
) -> Iterator[list[Path]]:
    """
    Sliding-window generator over `frames`.

    Step size is (batch_size - overlap), so consecutive batches share
    `overlap` frames at their boundary.

    Example with batch_size=4, overlap=1, frames=[0,1,2,3,4,5,6,7]:
        batch 0: [0, 1, 2, 3]
        batch 1: [3, 4, 5, 6]   ← frame 3 repeated
        batch 2: [6, 7]          ← frame 6 repeated, last batch is short
    """
    step  = batch_size - overlap
    start = 0
    while start < len(frames):
        yield frames[start : start + batch_size]
        start += step
