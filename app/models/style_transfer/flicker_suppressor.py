# ==============================================================================
# app/models/style_transfer/flicker_suppressor.py
#
# Post-processing pass that eliminates per-frame diffusion flickering.
#
# Why flickering happens:
#   Even with flow_guidance providing a good temporal init, diffusion is
#   stochastic — random noise is added and removed during inference.  At
#   motion boundaries and occlusion edges, the flow warping is ambiguous
#   (two surfaces overlap; which one "owns" the pixel?).  In these regions
#   the diffusion model makes different colour/texture choices on consecutive
#   frames, producing visible strobing.
#
# What flicker_suppressor does:
#   After the style engine runs, for each frame t we have:
#     new_styled[t]    — raw diffusion output (correct style, may flicker)
#     warped_prev[t]   — styled[t-1] warped forward (temporally stable,
#                        may have slightly wrong style in moved areas)
#
#   We blend them with a per-pixel weight α:
#     output[t] = α[y,x] * new_styled[t] + (1-α[y,x]) * warped_prev[t]
#
#   Where α = consistency_score(fwd_flow, bwd_flow):
#     α ≈ 1  →  pixel displacement is cycle-consistent (static background)
#                trust the new diffusion output entirely
#     α ≈ 0  →  pixel is on a motion boundary or occlusion
#                blend toward the warped previous frame (temporally stable)
#
# Net effect:
#   Static background areas get the full, correctly-styled diffusion output.
#   Moving object edges — where flickering is most visible — get blended
#   with the temporally-stable warped previous frame, eliminating the strobe.
#
# Relationship to flow_guidance.py:
#   flow_guidance   →  prevents flickering BEFORE diffusion  (better init)
#   flicker_suppressor →  removes residual flickering AFTER diffusion (post-process)
#   Together they are complementary; neither alone is as effective as both.
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

async def suppress(
    styled_frames:  list[np.ndarray],
    source_frames:  list[np.ndarray],
    flow_engine:    RAFTFlowEngine,
    prev_styled:    Optional[np.ndarray] = None,
    alpha_floor:    float = 0.3,
    alpha_ceil:     float = 1.0,
    blur_score:     bool  = True,
) -> list[np.ndarray]:
    """
    Apply flow-consistency-based flicker suppression to styled frames.

    Args:
        styled_frames:  Raw output from style_engine.transfer() — may flicker
        source_frames:  Original BGR frames (used for RAFT flow computation)
        flow_engine:    Shared RAFTFlowEngine instance
        prev_styled:    Styled frame from the previous batch (cross-batch
                        suppression at the first frame of this batch).
                        None → frame 0 is returned as-is.
        alpha_floor:    Minimum blend weight toward new styled frame.
                        0.0 = allow full suppression in inconsistent regions.
                        0.3 = always keep at least 30% of new styled output
                              (prevents over-smearing on fast motion).
        alpha_ceil:     Maximum blend weight (1.0 = full new styled output
                        in perfectly consistent regions).
        blur_score:     If True, apply Gaussian blur to the consistency score
                        before using it as blend weight.  This prevents sharp
                        transitions between blended and unblended regions.

    Returns:
        List of suppressed uint8 BGR frames, same length as styled_frames.
    """
    if not styled_frames:
        return []

    n        = len(styled_frames)
    output   = [None] * n

    # Frame 0: no previous styled frame in this batch → pass through unchanged
    # (unless prev_styled from the previous batch is available)
    if prev_styled is None:
        output[0] = styled_frames[0]
        start_t   = 1
    else:
        start_t   = 0

    for t in range(start_t, n):
        # The "previous styled" reference for frame t
        if t == 0:
            prev_ref = prev_styled          # cross-batch reference
        else:
            prev_ref = output[t - 1]        # previous suppressed frame (not raw styled)

        # ── Compute RAFT flow for this frame pair ─────────────────────────────
        fwd_flow, bwd_flow = await flow_engine.compute_flow_async(
            source_frames[t - 1] if t > 0 else source_frames[0],
            source_frames[t],
        )

        # ── Warp previous styled frame forward ────────────────────────────────
        warped_prev = flow_utils.warp_frame(
            prev_ref.astype(np.float32),
            fwd_flow,
            border_mode = cv2.BORDER_REFLECT_101,
        )
        warped_prev = np.clip(warped_prev, 0, 255).astype(np.uint8)

        # ── Compute per-pixel consistency score ───────────────────────────────
        #
        # score[y,x] ≈ 1  →  pixel (x,y) has consistent forward+backward flow
        #                     → static background → trust new styled output
        # score[y,x] ≈ 0  →  inconsistent flow (motion edge, occlusion)
        #                     → blend toward warped_prev (temporally stable)
        #
        score = flow_utils.consistency_score(fwd_flow, bwd_flow)   # (H, W) float32

        # ── Optional: blur the score for smooth spatial transitions ───────────
        if blur_score:
            score = _blur_score(score)

        # ── Clamp to [alpha_floor, alpha_ceil] ────────────────────────────────
        score = np.clip(score, alpha_floor, alpha_ceil)

        # ── Blend: α * new_styled + (1-α) * warped_prev ───────────────────────
        output[t] = flow_utils.blend_frames(
            frame_a = styled_frames[t],    # new diffusion output (may flicker)
            frame_b = warped_prev,         # temporally stable reference
            alpha   = score,               # per-pixel blend weight
        )

    if output[0] is None:
        output[0] = styled_frames[0]

    logger.debug(
        "flicker_suppressor: processed %d frames  "
        "alpha=[%.2f, %.2f]  blur=%s",
        n, alpha_floor, alpha_ceil, blur_score,
    )
    return output


def suppress_sync(
    styled_frames: list[np.ndarray],
    source_frames: list[np.ndarray],
    flow_engine:   RAFTFlowEngine,
    prev_styled:   Optional[np.ndarray] = None,
    **kwargs,
) -> list[np.ndarray]:
    """Synchronous wrapper — for use inside asyncio.to_thread contexts."""
    return asyncio.run(
        suppress(styled_frames, source_frames, flow_engine, prev_styled, **kwargs)
    )


# ── Score post-processing ──────────────────────────────────────────────────────

def _blur_score(score: np.ndarray, kernel_size: int = 15, sigma: float = 5.0) -> np.ndarray:
    """
    Spatially smooth the consistency score with a Gaussian kernel.

    Without blurring, the score map has hard edges at motion boundaries
    which would appear as visible seams in the blended output.
    A large sigma (5.0) creates a wide feather zone so the transition
    between "trust new output" and "use warped prev" is gradual.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(score, (kernel_size, kernel_size), sigma)
    return blurred.astype(np.float32)


# ── Diagnostics (for testing and debugging) ────────────────────────────────────

def score_to_heatmap(score: np.ndarray) -> np.ndarray:
    """
    Convert a consistency score (H, W) float32 in [0,1] to a BGR heatmap
    for visualisation.  Blue = consistent (α≈1), Red = inconsistent (α≈0).
    Useful for debugging which regions are being suppressed.
    """
    score_u8 = (score * 255).astype(np.uint8)
    return cv2.applyColorMap(score_u8, cv2.COLORMAP_JET)
