# ==============================================================================
# app/pipeline/orchestrator.py
#
# Master async pipeline runner — wires all 4 stages and emits SSE events.
#
# Current state after Step 4:
#   Stages 1 (extract), 2 (batch), and 4 (recompose) are fully implemented.
#   Stage 3 (infer) is still a stub that sleeps and emits progress events
#   without running any models.  It will be replaced in Steps 6 (Tool 1)
#   and 7 (Tool 2).
#
# Temporal consistency hooks — both gated by params flags sent from the UI:
#
#   Tool 1 — mask_propagator.py (Step 6):
#     if params["temporal_smoothing"]:
#         # flow_engine is the shared RAFT instance (Step 5)
#         masks = mask_propagator.propagate(masks, frames, flow_engine)
#
#   Tool 2 — flow_guidance.py + flicker_suppressor.py (Step 7):
#     if params["flow_guidance"]:
#         styled = flow_guidance.condition(styled, frames, flow_engine)
#     if params["flicker_suppression"]:
#         styled = flicker_suppressor.blend(styled, frames, flow_engine)
#
#   Both tools receive the same RAFTFlowEngine instance created once here
#   in _run() so identical flow fields are used for mask warping and style
#   conditioning — essential for cross-tool temporal consistency.
# ==============================================================================

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import settings
from app.core.job_store import job_store
from app.pipeline import batcher, extractor, recomposer
from app.schemas.job import JobStage

logger = logging.getLogger(__name__)


async def run_pipeline(
    job_id:        str,
    tool:          str,
    params:        Dict[str, Any],
    use_replicate: bool = False,
) -> None:
    """
    Entry point called by BackgroundTasks in process.py.
    Any unhandled exception is caught and emitted as an SSE error event.
    """
    try:
        await _run(job_id, tool, params, use_replicate)
    except Exception as exc:
        logger.exception("Pipeline failed for job %s: %s", job_id, exc)
        await job_store.update(job_id, stage=JobStage.ERROR, error=str(exc))
        await job_store.emit(job_id, {
            "event":   "error",
            "message": f"Pipeline error: {exc}",
        })
    finally:
        await job_store.close_stream(job_id)


# ── Inner pipeline ─────────────────────────────────────────────────────────────

async def _run(
    job_id:        str,
    tool:          str,
    params:        Dict[str, Any],
    use_replicate: bool,
) -> None:

    job         = await job_store.get(job_id)
    upload_path = _find_upload(job_id)

    # Convenience wrapper: update job state + emit SSE progress in one call
    async def progress(pct: int, message: str, detail: str = "") -> None:
        await job_store.update(job_id, progress=pct, message=message)
        await job_store.emit(job_id, {
            "event":   "progress",
            "stage":   (await job_store.get(job_id)).stage.value,
            "pct":     pct,
            "message": message,
            "detail":  detail,
        })

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 1 — Extract  (extractor.py — fully implemented in Step 4)
    # ═══════════════════════════════════════════════════════════════════════
    await job_store.update(job_id, stage=JobStage.EXTRACT)

    async def extract_cb(pct: int, message: str, detail: str = "") -> None:
        await job_store.update(job_id, progress=pct, message=message)
        await job_store.emit(job_id, {
            "event": "progress", "stage": "extract",
            "pct": pct, "message": message, "detail": detail,
        })

    frames_dir, audio_path, metadata = await extractor.extract(
        upload_path = upload_path,
        job_id      = job_id,
        progress_cb = extract_cb,
    )

    # Persist metadata back onto the job (useful for the export bar display)
    await job_store.update(
        job_id,
        fps        = metadata["fps"],
        width      = metadata["width"],
        height     = metadata["height"],
        duration_s = metadata["duration_s"],
    )

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 2 — Batch  (batcher.py — fully implemented in Step 4)
    # ═══════════════════════════════════════════════════════════════════════
    await job_store.update(job_id, stage=JobStage.BATCH)
    await job_store.emit(job_id, {
        "event": "progress", "stage": "batch", "pct": 11,
        "message": "Organising frame batches…",
        "detail": f"Batch size: {settings.infer_batch_size} frames",
    })

    # Tool 1 (ProPainter) benefits from 2-frame overlap between batches to
    # compute inter-batch optical flow without boundary artefacts.
    # Tool 2 (SVD/ControlNet) handles continuity at the pixel level via RAFT
    # so overlap is not needed at the batch level.
    overlap = 2 if tool == "object_removal" else 0

    batches = batcher.make_batches(
        frames_dir = frames_dir,
        batch_size = settings.infer_batch_size,
        overlap    = overlap,
    )
    total_frames_out = batcher.total_output_frames(batches, overlap)

    await job_store.emit(job_id, {
        "event": "progress", "stage": "batch", "pct": 15,
        "message": "Batches ready",
        "detail": f"{len(batches)} batches · {total_frames_out} output frames",
    })
    await job_store.update(job_id, progress=15)

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 3 — Infer  (Steps 6 + 7 replace this stub)
    #
    # Shared RAFT instance — created once, passed to both tools:
    #   from app.pipeline.temporal.optical_flow import RAFTFlowEngine
    #   flow_engine = RAFTFlowEngine(settings.raft_checkpoint)
    #
    # Tool 1 dispatch (Step 6):
    #   processed_frames = await _run_object_removal(
    #       batches, frames_dir, params, flow_engine, job_id, progress)
    #
    # Tool 2 dispatch (Step 7):
    #   processed_frames = await _run_style_transfer(
    #       batches, frames_dir, params, flow_engine, job_id, progress)
    # ═══════════════════════════════════════════════════════════════════════
    await job_store.update(job_id, stage=JobStage.INFER)
    processed_dir = Path(settings.processed_dir) / job_id
    processed_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(batches):
        pct     = 16 + int(((i + 1) / len(batches)) * 73)   # 16 → 89
        frames_done = (i + 1) * settings.infer_batch_size
        await job_store.emit(job_id, {
            "event":   "progress",
            "stage":   "infer",
            "pct":     pct,
            "message": f"Processing frame {min(frames_done, total_frames_out)} / {total_frames_out}",
            "detail":  f"[stub — Step 6/7] batch {i+1}/{len(batches)} · tool: {tool}",
        })
        await job_store.update(job_id, progress=pct)

        # ── STUB: copy input frames to processed/ unchanged ──────────────────
        # Steps 6 and 7 replace this loop body with real model inference.
        # The copy ensures recomposer.py has actual PNG files to work with
        # so the whole pipeline can be tested end-to-end before models land.
        import shutil
        keep = batcher.output_slice(i, batch, overlap)
        for frame_path in keep:
            dst = processed_dir / frame_path.name
            shutil.copy2(frame_path, dst)

        await asyncio.sleep(0.05)   # yield event loop so SSE events flush

    await job_store.update(job_id, progress=89)

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 4 — Recompose  (recomposer.py — fully implemented in Step 4)
    # ═══════════════════════════════════════════════════════════════════════
    await job_store.update(job_id, stage=JobStage.RECOMPOSE)

    output_path = Path(settings.output_dir) / f"{job_id}.mp4"

    async def recompose_cb(pct: int, message: str, detail: str = "") -> None:
        await job_store.update(job_id, progress=pct, message=message)
        await job_store.emit(job_id, {
            "event": "progress", "stage": "recompose",
            "pct": pct, "message": message, "detail": detail,
        })

    final_path = await recomposer.recompose(
        processed_dir = processed_dir,
        audio_path    = audio_path,
        output_path   = output_path,
        fps           = metadata["fps"],
        progress_cb   = recompose_cb,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # COMPLETE
    # ═══════════════════════════════════════════════════════════════════════
    await job_store.update(
        job_id,
        stage       = JobStage.COMPLETE,
        progress    = 100,
        output_path = str(final_path),
    )
    await job_store.emit(job_id, {
        "event":       "complete",
        "output_path": str(final_path),
        "duration_s":  metadata["duration_s"],
        "tool":        tool,
    })

    logger.info(
        "Pipeline COMPLETE: job=%s  tool=%s  output=%s  (%.1f MB)",
        job_id, tool, final_path,
        final_path.stat().st_size / 1_048_576 if final_path.exists() else 0,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_upload(job_id: str) -> Path:
    """
    Locate the uploaded file for a job — supports .mp4, .mov, and .webm.
    The upload route preserves the original extension, so we probe all three.
    """
    job_dir = Path(settings.upload_dir) / job_id
    for suffix in (".mp4", ".mov", ".webm"):
        candidate = job_dir / f"original{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No uploaded video found in {job_dir}. "
        "Expected original.mp4, original.mov, or original.webm."
    )
