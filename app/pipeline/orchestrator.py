# ==============================================================================
# app/pipeline/orchestrator.py
#
# Master async pipeline runner — wires all 4 stages and emits SSE events.
#
# Current state after Step 6:
#   Stage 1 (extract), Stage 2 (batch), Stage 4 (recompose) — fully wired
#   Stage 3 Tool 1 (object_removal) — fully wired
#   Stage 3 Tool 2 (style_transfer)  — stub (Steps 7 wires this)
# ==============================================================================

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict

from app.core.config import settings
from app.core.job_store import job_store
from app.pipeline import batcher, extractor, recomposer
from app.pipeline.temporal.optical_flow import RAFTFlowEngine
from app.schemas.job import JobStage

logger = logging.getLogger(__name__)


async def run_pipeline(
    job_id:        str,
    tool:          str,
    params:        Dict[str, Any],
    use_replicate: bool = False,
) -> None:
    try:
        await _run(job_id, tool, params, use_replicate)
    except Exception as exc:
        logger.exception("Pipeline failed for job %s: %s", job_id, exc)
        await job_store.update(job_id, stage=JobStage.ERROR, error=str(exc))
        await job_store.emit(job_id, {"event": "error", "message": f"Pipeline error: {exc}"})
    finally:
        await job_store.close_stream(job_id)


async def _run(
    job_id:        str,
    tool:          str,
    params:        Dict[str, Any],
    use_replicate: bool,
) -> None:

    upload_path  = _find_upload(job_id)
    processed_dir = Path(settings.processed_dir) / job_id
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── SSE progress helper ────────────────────────────────────────────────────
    async def progress(stage_val: str, pct: int, message: str, detail: str = "") -> None:
        await job_store.update(job_id, progress=pct, message=message)
        await job_store.emit(job_id, {
            "event": "progress", "stage": stage_val,
            "pct": pct, "message": message, "detail": detail,
        })

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Extract
    # ═══════════════════════════════════════════════════════════════════════════
    await job_store.update(job_id, stage=JobStage.EXTRACT)

    async def extract_cb(pct, msg, detail=""): await progress("extract", pct, msg, detail)

    frames_dir, audio_path, metadata = await extractor.extract(
        upload_path=upload_path, job_id=job_id, progress_cb=extract_cb,
    )
    await job_store.update(
        job_id,
        fps=metadata["fps"], width=metadata["width"],
        height=metadata["height"], duration_s=metadata["duration_s"],
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Batch
    # ═══════════════════════════════════════════════════════════════════════════
    await job_store.update(job_id, stage=JobStage.BATCH)
    await progress("batch", 11, "Organising frame batches…",
                   f"Batch size: {settings.infer_batch_size} frames")

    overlap = 2 if tool == "object_removal" else 0
    batches = batcher.make_batches(
        frames_dir=frames_dir,
        batch_size=settings.infer_batch_size,
        overlap=overlap,
    )
    total_out = batcher.total_output_frames(batches, overlap)
    await progress("batch", 15, "Batches ready",
                   f"{len(batches)} batches · {total_out} output frames")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Infer
    # ═══════════════════════════════════════════════════════════════════════════
    await job_store.update(job_id, stage=JobStage.INFER)

    # Create shared RAFT engine once — used by BOTH tools
    flow_engine = RAFTFlowEngine(settings.raft_checkpoint)

    if tool == "object_removal":
        await _infer_object_removal(
            job_id, batches, overlap, total_out, processed_dir,
            params, flow_engine, use_replicate, progress,
        )
    else:
        await _infer_style_transfer(
            job_id, batches, overlap, total_out, processed_dir,
            params, flow_engine, use_replicate, progress,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 4 — Recompose
    # ═══════════════════════════════════════════════════════════════════════════
    await job_store.update(job_id, stage=JobStage.RECOMPOSE)

    async def recompose_cb(pct, msg, detail=""): await progress("recompose", pct, msg, detail)

    output_path = Path(settings.output_dir) / f"{job_id}.mp4"
    final_path  = await recomposer.recompose(
        processed_dir=processed_dir,
        audio_path=audio_path,
        output_path=output_path,
        fps=metadata["fps"],
        progress_cb=recompose_cb,
    )

    await job_store.update(job_id, stage=JobStage.COMPLETE, progress=100, output_path=str(final_path))
    await job_store.emit(job_id, {
        "event":       "complete",
        "output_path": str(final_path),
        "duration_s":  metadata["duration_s"],
        "tool":        tool,
    })
    logger.info("Pipeline COMPLETE  job=%s  tool=%s  output=%s", job_id, tool, final_path)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — Object Removal
# ══════════════════════════════════════════════════════════════════════════════

async def _infer_object_removal(
    job_id:        str,
    batches:       list,
    overlap:       int,
    total_out:     int,
    processed_dir: Path,
    params:        Dict[str, Any],
    flow_engine:   RAFTFlowEngine,
    use_replicate: bool,
    progress,
) -> None:
    """
    Per-batch pipeline for Tool 1:
      1. Read frames from disk
      2. Segment the first frame of the batch (SAM / GroundingDINO)
      3. Propagate mask across all frames via RAFT (if temporal_smoothing=True)
      4. Inpaint the masked region (ProPainter / E2FGVI / Replicate)
      5. Write output frames to processed_dir

    The mask for the last frame of each batch is carried forward to the next
    batch as the "initial_mask" so SAM only runs once (or every
    RE_SEGMENT_INTERVAL frames for drift correction) for the entire video.
    """
    from app.models.object_removal import inpainter, mask_propagator, segmenter

    prompt            = params.get("prompt", "")
    detection_backend = params.get("detection_backend", "grounding_dino")
    inpaint_model     = params.get("inpaint_model", "propainter")
    temporal_smooth   = params.get("temporal_smoothing", True)
    dilation_px       = params.get("mask_dilation_px", 8)

    frames_written = 0
    carry_mask     = None   # mask propagated from last frame of previous batch

    for batch_idx, batch_paths in enumerate(batches):
        pct = 16 + int(((batch_idx + 1) / len(batches)) * 73)
        batch_num_label = f"batch {batch_idx+1}/{len(batches)}"

        # ── Load frames ───────────────────────────────────────────────────────
        import cv2 as _cv2
        frames = [_cv2.imread(str(p)) for p in batch_paths]

        # ── Segment first frame (or use carried mask) ─────────────────────────
        await progress("infer", max(16, pct - 2),
                       f"Segmenting frame {frames_written+1}/{total_out}",
                       f"{batch_num_label} · GroundingDINO + SAM")

        if carry_mask is None:
            # Very first frame — run full segmentation
            initial_mask = await _run_in_thread(
                segmenter.segment,
                frames[0], prompt, None, detection_backend, dilation_px,
            )
        else:
            # Use the carried mask from the previous batch's last frame
            initial_mask = carry_mask

        # ── Propagate mask across batch ───────────────────────────────────────
        if temporal_smooth and len(frames) > 1:
            await progress("infer", pct,
                           f"Propagating mask  {batch_num_label}",
                           "RAFT optical flow mask warping")
            masks = await mask_propagator.propagate(
                initial_mask      = initial_mask,
                frames            = frames,
                flow_engine       = flow_engine,
                dilation_px       = dilation_px,
                resegment         = True,
                prompt            = prompt,
                bbox              = None,
                detection_backend = detection_backend,
            )
        else:
            # temporal_smoothing=False → same mask for every frame (fast but less accurate)
            masks = [initial_mask] * len(frames)

        # Carry last mask to next batch (covers the overlap frames)
        carry_mask = masks[-1]

        # ── Inpaint ───────────────────────────────────────────────────────────
        await progress("infer", pct,
                       f"Inpainting  {batch_num_label}",
                       f"model: {inpaint_model}")
        output_frames = await inpainter.inpaint(
            frames        = frames,
            masks         = masks,
            model_type    = inpaint_model,
            use_replicate = use_replicate,
        )

        # ── Write output frames ───────────────────────────────────────────────
        keep_paths    = batcher.output_slice(batch_idx, batch_paths, overlap)
        keep_frames   = output_frames[:len(keep_paths)]

        for frame, src_path in zip(keep_frames, keep_paths):
            out_p = processed_dir / src_path.name
            _cv2.imwrite(str(out_p), frame)
            frames_written += 1

        await progress("infer", pct,
                       f"Processing frame {frames_written}/{total_out}",
                       f"{batch_num_label} complete")

    logger.info("Tool 1 inference done: %d frames written to %s", frames_written, processed_dir)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Style Transfer STUB (Step 7 replaces this)
# ══════════════════════════════════════════════════════════════════════════════

async def _infer_style_transfer_stub(
    job_id, batches, overlap, total_out, processed_dir,
    params, flow_engine, use_replicate, progress,
) -> None:
    """
    Stub: copies input frames unchanged to processed_dir.
    Step 7 replaces this with style_engine → flow_guidance → flicker_suppressor.
    """
    import asyncio as _asyncio
    import cv2 as _cv2

    frames_written = 0
    for batch_idx, batch_paths in enumerate(batches):
        pct = 16 + int(((batch_idx + 1) / len(batches)) * 73)
        keep = batcher.output_slice(batch_idx, batch_paths, overlap)
        for p in keep:
            dst = processed_dir / p.name
            shutil.copy2(p, dst)
            frames_written += 1
        await progress("infer", pct,
                       f"Processing frame {frames_written}/{total_out}",
                       "[stub] style_transfer — Step 7 pending")
        await _asyncio.sleep(0.03)


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _run_in_thread(fn, *args, **kwargs):
    """Run a blocking function in a thread executor."""
    import asyncio
    return await asyncio.to_thread(fn, *args, **kwargs)


def _find_upload(job_id: str) -> Path:
    job_dir = Path(settings.upload_dir) / job_id
    for suffix in (".mp4", ".mov", ".webm"):
        candidate = job_dir / f"original{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No uploaded video found in {job_dir}. "
        "Expected original.mp4, original.mov, or original.webm."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Style Transfer (Step 7)
# ══════════════════════════════════════════════════════════════════════════════

async def _infer_style_transfer(
    job_id:        str,
    batches:       list,
    overlap:       int,
    total_out:     int,
    processed_dir: Path,
    params:        dict,
    flow_engine:   RAFTFlowEngine,
    use_replicate: bool,
    progress,
) -> None:
    """
    Per-batch pipeline for Tool 2:
      1. Read frames from disk
      2. Build flow-guided init images (flow_guidance.py)
      3. Run diffusion style transfer (style_engine.py)
      4. Suppress residual flickering (flicker_suppressor.py)
      5. Write output frames to processed_dir

    Cross-batch state:
      carry_styled   — last styled+suppressed frame from prev batch.
                       Used as init for first frame of next batch (flow_guidance)
                       and as prev reference for first frame suppression.
      carry_source   — last SOURCE frame from prev batch.
                       Used as the "t-1" frame for RAFT when computing flow
                       to the first frame of the next batch.
    """
    from app.models.style_transfer import (
        flicker_suppressor,
        flow_guidance,
        style_engine,
    )

    prompt       = params.get("prompt", "")
    backbone     = params.get("style_backbone", "svd")
    strength     = params.get("style_strength", 0.75)
    steps        = params.get("inference_steps", 25)
    use_flow_g   = params.get("flow_guidance", True)
    use_flicker  = params.get("flicker_suppression", True)

    frames_written = 0
    carry_styled: dict | None = None   # {"frame": np.ndarray, "source": np.ndarray}

    for batch_idx, batch_paths in enumerate(batches):
        pct        = 16 + int(((batch_idx + 1) / len(batches)) * 73)
        batch_label = f"batch {batch_idx+1}/{len(batches)}"

        import cv2 as _cv2
        frames = [_cv2.imread(str(p)) for p in batch_paths]

        # ── Step A: Build flow-guided init images ─────────────────────────────
        init_frames = None
        if use_flow_g:
            await progress("infer", max(16, pct - 3),
                           f"Computing flow guidance  {batch_label}",
                           "RAFT flow → warp prev styled frame")
            prev_s = carry_styled["frame"]  if carry_styled else None
            init_frames = await flow_guidance.build_init_frames(
                source_frames = frames,
                flow_engine   = flow_engine,
                prev_styled   = prev_s,
                styled_frames = None,
            )

        # ── Step B: Style transfer ────────────────────────────────────────────
        await progress("infer", pct,
                       f"Styling frames  {batch_label}",
                       f"backbone: {backbone}  prompt: '{prompt[:40]}'")
        raw_styled = await style_engine.transfer(
            source_frames = frames,
            prompt        = prompt,
            strength      = strength,
            steps         = steps,
            backbone      = backbone,
            use_replicate = use_replicate,
            init_frames   = init_frames,
        )

        # ── Step C: Flicker suppression ───────────────────────────────────────
        if use_flicker:
            await progress("infer", min(pct + 2, 89),
                           f"Suppressing flicker  {batch_label}",
                           "consistency_score → per-pixel blend")
            prev_s = carry_styled["frame"]  if carry_styled else None
            output_frames = await flicker_suppressor.suppress(
                styled_frames = raw_styled,
                source_frames = frames,
                flow_engine   = flow_engine,
                prev_styled   = prev_s,
            )
        else:
            output_frames = raw_styled

        # ── Update cross-batch carry state ────────────────────────────────────
        carry_styled = {
            "frame":  output_frames[-1],
            "source": frames[-1],
        }

        # ── Write output frames ───────────────────────────────────────────────
        keep_paths  = batcher.output_slice(batch_idx, batch_paths, overlap)
        keep_frames = output_frames[:len(keep_paths)]

        for frame, src_path in zip(keep_frames, keep_paths):
            out_p = processed_dir / src_path.name
            _cv2.imwrite(str(out_p), frame)
            frames_written += 1

        await progress("infer", pct,
                       f"Processing frame {frames_written}/{total_out}",
                       f"{batch_label} complete")

    logger.info("Tool 2 inference done: %d frames written to %s", frames_written, processed_dir)
