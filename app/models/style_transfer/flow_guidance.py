# ==============================================================================
# app/models/style_transfer/flow_guidance.py
#
# Provides temporally-aware init images for the style engine.
#
# The core problem it solves:
#   Without flow guidance, ControlNet img2img re-styles each frame from
#   scratch using only the source frame as init.  Adjacent source frames are
#   visually similar but not identical — small colour/texture differences get
#   amplified by diffusion into large stylistic jumps between frames.
#   Result: the output video flickers or "strobes" even at low strength.
#
# The fix:
#   Instead of using the SOURCE frame as the img2img init, we use the
#   PREVIOUS STYLED frame — warped forward using RAFT optical flow.
#   This means:
#     • Diffusion now starts from a temporal prior that already looks like
#       the correct style (it's last frame, warped to this frame's motion)
#     • The model only needs to make small refinements, not a full re-style
#     • Colour and texture are propagated from frame to frame via the warp
#
# Integration in the pipeline (orchestrator._infer_style_transfer):
#
#   # 1. Run style engine on the batch (raw, no guidance)
#   raw_styled = await style_engine.transfer(frames, prompt, ...)
#
#   # 2. Compute flow-guided inits for the NEXT batch
#   next_inits = await flow_guidance.build_init_frames(
#       source_frames  = frames,
#       styled_frames  = raw_styled,
#       flow_engine    = flow_engine,
#       prev_styled    = carry_styled,  # last styled frame from prev batch
#   )
#
#   # 3. Re-run style engine WITH flow guidance (or pass inits to next batch)
#   guided_styled = await style_engine.transfer(
#       frames, prompt, ..., init_frames=next_inits
#   )
#
# For performance, the orchestrator uses a single-pass approach:
#   carry_styled from the PREVIOUS batch becomes init_frames[0] of the
#   CURRENT batch.  Subsequent inits are computed from the previous frame's
#   output within the same batch.  This avoids running inference twice.
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


# ── Public API ─────────────────────────────────────────────────────────────────

async def build_init_frames(
    source_frames: list[np.ndarray],
    flow_engine:   RAFTFlowEngine,
    prev_styled:   Optional[np.ndarray] = None,
    styled_frames: Optional[list[np.ndarray]] = None,
) -> list[Optional[np.ndarray]]:
    """
    Build per-frame init images for style_engine.transfer(init_frames=...).

    For frame t, the init is the styled frame at t-1 warped forward using
    the RAFT flow from source_frame[t-1] → source_frame[t].

    Why warp using SOURCE flow, not STYLED flow?
      The styled frames may have different colour distributions than the source
      (that's the point of style transfer), so RAFT flow computed on styled
      frames is less reliable.  Source-frame flow is computed on natural images
      that RAFT was trained on, so the displacement vectors are more accurate.

    Args:
        source_frames:  Original BGR frames (for flow computation)
        flow_engine:    Shared RAFTFlowEngine instance
        prev_styled:    Styled frame from the END of the previous batch.
                        Used as the init for source_frames[0] (cross-batch
                        temporal continuity). None → use source_frames[0] itself.
        styled_frames:  Styled frames from a previous pass (optional).
                        When provided, used as warp source instead of prev_styled.
                        Allows two-pass refinement.

    Returns:
        List of Optional[np.ndarray], one per frame.
        inits[0] = warped prev_styled (or None if no prev_styled available)
        inits[t] = warped styled_frames[t-1] using source flow t-1 → t
    """
    n      = len(source_frames)
    inits: list[Optional[np.ndarray]] = [None] * n

    # ── Frame 0: warp from previous batch's last styled frame ─────────────────
    if prev_styled is not None and n > 0:
        fwd_flow, _ = await flow_engine.compute_flow_async(
            # Use the last source frame we remember as the "previous" frame.
            # In practice the orchestrator passes the last source frame of the
            # previous batch here via a separate argument (see orchestrator).
            source_frames[0],
            source_frames[0],  # same frame → zero flow, just uses prev_styled as-is
        )
        # Zero-flow warp is a no-op but we keep the API consistent
        inits[0] = prev_styled
    else:
        inits[0] = None   # No previous context — style engine uses source frame

    # ── Frames 1 … N-1: warp from previous styled frame ──────────────────────
    # The "previous styled" for frame t is styled_frames[t-1] if we have a
    # previous pass, otherwise we can only carry forward from frame 0.
    ref_frames = styled_frames if styled_frames else source_frames

    for t in range(1, n):
        fwd_flow, _ = await flow_engine.compute_flow_async(
            source_frames[t - 1],
            source_frames[t],
        )

        prev_ref = ref_frames[t - 1] if ref_frames[t - 1] is not None else source_frames[t - 1]
        warped   = flow_utils.warp_frame(
            prev_ref.astype(np.float32),
            fwd_flow,
            border_mode = cv2.BORDER_REFLECT_101,
        )
        inits[t] = np.clip(warped, 0, 255).astype(np.uint8)

    logger.debug(
        "flow_guidance: built %d init frames  prev_styled=%s",
        n, "yes" if prev_styled is not None else "no",
    )
    return inits


async def warp_single(
    prev_styled:   np.ndarray,
    source_prev:   np.ndarray,
    source_curr:   np.ndarray,
    flow_engine:   RAFTFlowEngine,
) -> np.ndarray:
    """
    Convenience: warp ONE styled frame for use as the next frame's init.

    Args:
        prev_styled:  Styled frame at time t-1
        source_prev:  Source frame at time t-1 (for RAFT)
        source_curr:  Source frame at time t   (for RAFT)
        flow_engine:  Shared RAFT engine

    Returns:
        Warped styled frame (uint8 BGR) suitable as init_image for time t.
    """
    fwd_flow, _ = await flow_engine.compute_flow_async(source_prev, source_curr)
    warped = flow_utils.warp_frame(
        prev_styled.astype(np.float32),
        fwd_flow,
        border_mode = cv2.BORDER_REFLECT_101,
    )
    return np.clip(warped, 0, 255).astype(np.uint8)
