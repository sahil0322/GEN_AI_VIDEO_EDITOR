# ==============================================================================
# app/models/object_removal/mask_propagator.py
#
# Propagates the frame-0 segmentation mask across an entire batch of frames
# using RAFT optical flow.
#
# Why not just run SAM on every frame?
#   SAM ViT-H takes ~200 ms per frame on a GPU.  A 5-second clip at 30 fps
#   is 150 frames × 200 ms = 30 seconds just for segmentation.
#   RAFT flow + warp_frame() takes ~30 ms per frame pair — 10× faster.
#
# The propagation strategy:
#   1. Segment frame 0 with SAM → binary mask M_0
#   2. For each subsequent frame t:
#        a. Compute RAFT forward flow from frame_{t-1} to frame_t
#        b. Warp M_{t-1} using that flow field → M_t (float, soft)
#        c. Threshold + dilate → binary M_t
#   3. Every RE_SEGMENT_INTERVAL frames, re-run SAM to correct drift
#      (RAFT accumulates small errors over long sequences)
#
# Drift correction:
#   Over 60+ frames, small per-frame warp errors accumulate — the mask
#   gradually drifts away from the actual object.  Re-running SAM every
#   RE_SEGMENT_INTERVAL frames (default: 30) resets the drift without
#   paying the full SAM cost on every frame.
# ==============================================================================

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import cv2
import numpy as np

from app.pipeline.temporal import flow_utils
from app.pipeline.temporal.optical_flow import RAFTFlowEngine

logger = logging.getLogger(__name__)

# Re-run SAM this often to correct accumulated warp drift.
# 30 = every second at 30fps — balances quality vs speed.
# Set to 0 to disable (propagate only, fastest but may drift).
RE_SEGMENT_INTERVAL = 30


# ── Public API ─────────────────────────────────────────────────────────────────

async def propagate(
    initial_mask:  np.ndarray,
    frames:        list[np.ndarray],
    flow_engine:   RAFTFlowEngine,
    dilation_px:   int  = 8,
    resegment:     bool = True,
    prompt:        Optional[str]   = None,
    bbox:          Optional[tuple] = None,
    detection_backend: str         = "grounding_dino",
) -> list[np.ndarray]:
    """
    Propagate `initial_mask` across all frames in the batch.

    Args:
        initial_mask:    uint8 (H, W) mask from segmenter.segment() — 255/0
        frames:          list of uint8 BGR numpy arrays (H, W, 3) — full batch
        flow_engine:     shared RAFTFlowEngine instance from orchestrator._run()
        dilation_px:     re-dilate after each warp to keep boundary coverage
        resegment:       if True, re-run SAM every RE_SEGMENT_INTERVAL frames
        prompt/bbox/detection_backend: forwarded to segmenter.segment() for
                         drift-correction re-segmentations (ignored when
                         resegment=False or RE_SEGMENT_INTERVAL==0)

    Returns:
        List of uint8 (H, W) masks, one per frame in `frames`.
        masks[0] is the (dilated) initial_mask.
        masks[i] is the propagated mask for frames[i].
    """
    if not frames:
        return []

    # ── Frame 0: use the provided initial mask ─────────────────────────────────
    masks: list[np.ndarray] = [_post_process(initial_mask, dilation_px)]
    logger.info(
        "mask_propagator: propagating across %d frames  "
        "resegment=%s  interval=%d  dilation=%dpx",
        len(frames), resegment, RE_SEGMENT_INTERVAL, dilation_px,
    )

    # ── Frames 1 … N-1: warp forward from previous mask ───────────────────────
    for t in range(1, len(frames)):

        # Optional drift correction: re-run full segmentation
        if resegment and RE_SEGMENT_INTERVAL > 0 and t % RE_SEGMENT_INTERVAL == 0:
            corrected = _resegment(frames[t], prompt, bbox, detection_backend, dilation_px)
            if corrected is not None:
                masks.append(corrected)
                logger.debug("Drift correction via SAM at frame %d", t)
                continue

        # Normal path: warp previous mask using RAFT flow ─────────────────────
        fwd_flow, _ = await flow_engine.compute_flow_async(frames[t - 1], frames[t])

        # Warp as float so sub-pixel motion is handled smoothly
        prev_float  = masks[t - 1].astype(np.float32) / 255.0  # [0, 1]
        warped_float = flow_utils.warp_frame(
            prev_float,
            fwd_flow,
            interpolation = cv2.INTER_LINEAR,
            border_mode   = cv2.BORDER_CONSTANT,  # out-of-bounds → 0 (no mask)
        )

        # Threshold back to binary, then dilate
        binary = ((warped_float > 0.5).astype(np.uint8)) * 255
        masks.append(_post_process(binary, dilation_px))

    logger.info(
        "mask_propagator: done — %d masks  avg_coverage=%.1f%%",
        len(masks),
        100.0 * np.mean([m.sum() / m.size for m in masks]),
    )
    return masks


def propagate_sync(
    initial_mask:  np.ndarray,
    frames:        list[np.ndarray],
    flow_engine:   RAFTFlowEngine,
    dilation_px:   int  = 8,
    resegment:     bool = True,
    prompt:        Optional[str]   = None,
    bbox:          Optional[tuple] = None,
    detection_backend: str         = "grounding_dino",
) -> list[np.ndarray]:
    """
    Synchronous wrapper for propagate() — for use in thread-executor contexts
    (e.g. when the caller is already inside asyncio.to_thread).
    """
    return asyncio.run(
        propagate(
            initial_mask      = initial_mask,
            frames            = frames,
            flow_engine       = flow_engine,
            dilation_px       = dilation_px,
            resegment         = resegment,
            prompt            = prompt,
            bbox              = bbox,
            detection_backend = detection_backend,
        )
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _post_process(mask: np.ndarray, dilation_px: int) -> np.ndarray:
    """
    Clean up a raw binary mask:
      1. Ensure it's uint8 with values 0 / 255
      2. Remove speckling with morphological closing (fills small holes)
      3. Dilate to cover object boundaries
    """
    # Normalise to 0/255 uint8
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8) * 255

    # Morphological closing: fill small holes inside the mask
    # (SAM sometimes leaves gaps inside large objects)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

    # Dilation: expand boundary coverage
    if dilation_px > 0:
        mask = flow_utils.dilate_mask(mask, dilation_px)

    return mask


def _resegment(
    frame:            np.ndarray,
    prompt:           Optional[str],
    bbox:             Optional[tuple],
    detection_backend: str,
    dilation_px:      int,
) -> Optional[np.ndarray]:
    """
    Re-run segmenter.segment() on `frame` for drift correction.
    Returns None (silently) if the import or inference fails — in that
    case the caller will fall back to warp propagation.
    """
    if prompt is None and bbox is None:
        return None
    try:
        from app.models.object_removal.segmenter import segment
        mask = segment(
            frame             = frame,
            prompt            = prompt,
            bbox              = bbox,
            detection_backend = detection_backend,
            dilation_px       = dilation_px,
        )
        return mask
    except Exception as exc:
        logger.warning("Drift-correction SAM call failed: %s — using warp instead", exc)
        return None


# ── Mask quality helpers (used by inpainter.py) ───────────────────────────────

def masks_to_tensor_batch(masks: list[np.ndarray]) -> "torch.Tensor":
    """
    Convert a list of (H, W) uint8 masks to a float32 tensor [T, 1, H, W]
    in the range [0, 1].  Used by ProPainter and E2FGVI which expect this
    exact format as input.
    """
    import torch
    t = [torch.from_numpy(m.astype(np.float32) / 255.0).unsqueeze(0) for m in masks]
    return torch.stack(t, dim=0)   # [T, 1, H, W]


def frames_to_tensor_batch(frames: list[np.ndarray]) -> "torch.Tensor":
    """
    Convert a list of (H, W, 3) uint8 BGR frames to a float32 tensor
    [T, 3, H, W] in the range [0, 1] with RGB channel order.
    Used by ProPainter and E2FGVI.
    """
    import torch
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    t = [torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1) for f in rgb_frames]
    return torch.stack(t, dim=0)   # [T, 3, H, W]
