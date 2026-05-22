# ==============================================================================
# app/pipeline/orchestrator.py
#
# Master async pipeline runner.
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CURRENT STATE: Working stub (Step 3)                                   │
# │  The stub simulates all 4 pipeline stages with real timing delays so    │
# │  the entire SSE flow (frontend → /process → /status → EventSource)      │
# │  can be tested end-to-end before any AI models are wired in.            │
# │                                                                         │
# │  Step 8 replaces the stub bodies with real calls to:                    │
# │    extractor.py  →  batcher.py  →  [model dispatch]  →  recomposer.py  │
# └─────────────────────────────────────────────────────────────────────────┘
#
# Temporal consistency hooks (both are gated by params flags):
#
#   Tool 1 — mask_propagator.py:
#     if params["temporal_smoothing"]:
#         masks = mask_propagator.propagate(masks, frames, flow_engine)
#     where flow_engine is the RAFT instance from pipeline/temporal/optical_flow.py
#
#   Tool 2 — flow_guidance.py + flicker_suppressor.py:
#     if params["flow_guidance"]:
#         styled = flow_guidance.condition(styled, frames, flow_engine)
#     if params["flicker_suppression"]:
#         styled = flicker_suppressor.blend(styled, frames, flow_engine)
#
# Both tools share the same RAFTFlowEngine instance (created once per pipeline
# run and passed down) so identical flow fields are used for mask warping and
# style conditioning — a critical requirement for temporal consistency.
# ==============================================================================

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict

from app.core.config import settings
from app.core.job_store import job_store
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

    Runs the 4-stage pipeline and emits SSE events after each stage.
    Any unhandled exception is caught, logged, and emitted as an "error" event
    so the frontend always receives a terminal event (never hangs).

    Args:
        job_id:        8-char hex job identifier
        tool:          "object_removal" | "style_transfer"
        params:        validated dict from ProcessRequest.params
        use_replicate: True → offload heavy inference to Replicate API
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
        # Always close the stream so the SSE generator terminates cleanly
        await job_store.close_stream(job_id)


async def _run(
    job_id:        str,
    tool:          str,
    params:        Dict[str, Any],
    use_replicate: bool,
) -> None:
    """Inner pipeline — replace stub sections with real calls in Step 8."""

    job = await job_store.get(job_id)
    upload_path = _find_upload(job_id)

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Extract
    # Full implementation in Step 4: extractor.py
    #   frames_dir, audio_path = await extractor.extract(upload_path, job_id)
    # ═══════════════════════════════════════════════════════════════════════════
    await job_store.update(job_id, stage=JobStage.EXTRACT, progress=2)
    await _emit_progress(job_id, "extract", 2, "Extracting frames…", "Splitting video into PNG frames")

    # STUB: simulate extraction time
    await asyncio.sleep(1.5)

    frames_dir = Path(settings.frames_dir) / job_id
    audio_path = frames_dir / "audio.aac"

    await _emit_progress(job_id, "extract", 10, "Frames extracted", f"Extracted to {frames_dir}")
    await job_store.update(job_id, stage=JobStage.EXTRACT, progress=10)

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Batch
    # Full implementation in Step 4: batcher.py
    #   batches = batcher.make_batches(frames_dir, settings.infer_batch_size)
    # ═══════════════════════════════════════════════════════════════════════════
    await job_store.update(job_id, stage=JobStage.BATCH, progress=12)
    await _emit_progress(job_id, "batch", 12, "Organising batches…", f"Batch size: {settings.infer_batch_size} frames")

    # STUB: simulate batching
    await asyncio.sleep(0.5)

    stub_total_batches = 15  # placeholder
    await _emit_progress(job_id, "batch", 15, "Batches ready", f"{stub_total_batches} batches of {settings.infer_batch_size} frames")
    await job_store.update(job_id, stage=JobStage.BATCH, progress=15)

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Infer
    # Full implementation in Steps 6 & 7.
    #
    # Shared RAFT engine (Step 5):
    #   from app.pipeline.temporal.optical_flow import RAFTFlowEngine
    #   flow_engine = RAFTFlowEngine(settings.raft_checkpoint)
    #
    # Tool 1 dispatch:
    #   from app.models.object_removal.segmenter      import Segmenter
    #   from app.models.object_removal.mask_propagator import MaskPropagator
    #   from app.models.object_removal.inpainter       import Inpainter
    #
    # Tool 2 dispatch:
    #   from app.models.style_transfer.style_engine        import StyleEngine
    #   from app.models.style_transfer.flow_guidance       import FlowGuidance
    #   from app.models.style_transfer.flicker_suppressor  import FlickerSuppressor
    # ═══════════════════════════════════════════════════════════════════════════
    await job_store.update(job_id, stage=JobStage.INFER, progress=16)

    # STUB: simulate batch-by-batch inference progress
    for i in range(stub_total_batches):
        pct = 16 + int(((i + 1) / stub_total_batches) * 73)  # 16 → 89
        await _emit_progress(
            job_id, "infer", pct,
            f"Processing frame {(i+1)*settings.infer_batch_size} / {stub_total_batches*settings.infer_batch_size}",
            f"[stub] {tool} — batch {i+1} / {stub_total_batches}  (models not yet loaded)",
        )
        await job_store.update(job_id, progress=pct)
        await asyncio.sleep(0.4)  # simulate GPU time

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 4 — Recompose
    # Full implementation in Step 4: recomposer.py
    #   output_path = await recomposer.recompose(
    #       processed_dir=Path(settings.processed_dir) / job_id,
    #       audio_path=audio_path,
    #       output_path=Path(settings.output_dir) / f"{job_id}.mp4",
    #       fps=job.fps,
    #   )
    # ═══════════════════════════════════════════════════════════════════════════
    await job_store.update(job_id, stage=JobStage.RECOMPOSE, progress=90)
    await _emit_progress(job_id, "recompose", 90, "Stitching final video…", "Running FFmpeg recomposition")

    await asyncio.sleep(1.0)  # STUB

    await _emit_progress(job_id, "recompose", 99, "Muxing audio track…", "")
    await asyncio.sleep(0.5)  # STUB

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPLETE
    # ═══════════════════════════════════════════════════════════════════════════
    # STUB: no real output file exists yet — Step 8 sets the real path.
    stub_output = f"storage/outputs/{job_id}.mp4"

    await job_store.update(
        job_id,
        stage       = JobStage.COMPLETE,
        progress    = 100,
        output_path = stub_output,
    )
    await job_store.emit(job_id, {
        "event":       "complete",
        "output_path": stub_output,
        "duration_s":  (await job_store.get(job_id)).duration_s or 0,
        "tool":        tool,
    })

    logger.info("Pipeline COMPLETE (stub) for job %s  tool=%s", job_id, tool)


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _emit_progress(
    job_id:  str,
    stage:   str,
    pct:     int,
    message: str,
    detail:  str = "",
) -> None:
    """Convenience wrapper — constructs and emits a progress event dict."""
    await job_store.emit(job_id, {
        "event":   "progress",
        "stage":   stage,
        "pct":     pct,
        "message": message,
        "detail":  detail,
    })


def _find_upload(job_id: str) -> Path:
    """
    Locate the uploaded file for a job.
    Supports .mp4, .mov, and .webm (the extension is preserved from upload).
    Raises FileNotFoundError if nothing is found — the pipeline should not
    have been started if the upload failed.
    """
    job_dir = Path(settings.upload_dir) / job_id
    for suffix in (".mp4", ".mov", ".webm"):
        candidate = job_dir / f"original{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No uploaded video found in {job_dir}. "
        "Expected one of: original.mp4, original.mov, original.webm"
    )
