# ==============================================================================
# app/pipeline/temporal/flow_utils.py
#
# Flow field manipulation utilities — used by both AI tools.
#
# Three core functions and what they fix:
#
#  warp_frame()
#    Takes a frame and a flow field and produces a spatially remapped version
#    of that frame.  Used by:
#      • mask_propagator.py (Tool 1) to warp the segmentation mask forward
#        in time — avoids re-running SAM on every frame, prevents mask jitter.
#      • flow_guidance.py (Tool 2) to warp the previous styled frame so it
#        can be used as ControlNet conditioning for the next frame.
#
#  consistency_score()
#    Computes a per-pixel float score in [0, 1] measuring how much the forward
#    and backward flow fields agree (cycle-consistency).
#      score ≈ 1 → pixel displacement is consistent (static background)
#      score ≈ 0 → pixel is on a motion boundary or occlusion
#    Used by:
#      • flicker_suppressor.py (Tool 2) as the blend weight: stable pixels
#        take the new diffusion output; unstable pixels blend with the
#        warped-previous frame, preventing the strobing/flickering artefact.
#
#  blend_frames()
#    Per-pixel alpha blend of two frames.  Vectorised numpy, operates in
#    float32 and returns uint8 for immediate PNG save.
#
# All functions are pure numpy/OpenCV (no torch) so they work on CPU without
# a GPU and can be called from any thread without CUDA context issues.
# ==============================================================================

from __future__ import annotations

import cv2
import numpy as np


# ── Core public API ────────────────────────────────────────────────────────────

def warp_frame(
    frame: np.ndarray,
    flow:  np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode:   int = cv2.BORDER_REFLECT_101,
) -> np.ndarray:
    """
    Warp `frame` according to `flow` using backward-mapping remap.

    For each output pixel (x, y) we sample the input at (x + dx, y + dy),
    where (dx, dy) = flow[y, x].  This is "backward warping" — we are
    asking "where did this output pixel come from in the input?"

    Why backward warping?
      Forward warping (scattering each input pixel to its destination) creates
      holes where two input pixels map to non-adjacent output pixels.
      Backward warping (pulling from the source for each output pixel) avoids
      holes at the cost of requiring the backward flow field.

    Args:
        frame:        uint8 or float32 numpy array (H, W, C) or (H, W)
        flow:         float32 numpy array (H, W, 2)
                      flow[y, x, 0] = horizontal displacement (dx)
                      flow[y, x, 1] = vertical   displacement (dy)
        interpolation: OpenCV interpolation flag (INTER_LINEAR is a good default)
        border_mode:   How to handle pixels that map outside the frame boundary.
                       BORDER_REFLECT_101 avoids visible seams at edges.

    Returns:
        Warped frame with same shape and dtype as `frame`.

    Example (mask propagation in Tool 1):
        # frame_t is the binary mask at time t (0/1 float32)
        # fwd_flow is the flow from frame_t to frame_{t+1}
        mask_t1 = warp_frame(mask_t.astype("float32"), fwd_flow)
        mask_t1 = (mask_t1 > 0.5).astype("uint8")  # threshold back to binary
    """
    H, W = frame.shape[:2]

    # Build the absolute sampling coordinates: pixel_pos + displacement
    # meshgrid gives us (x, y) grid in the (W, H) range
    grid_x = np.arange(W, dtype=np.float32)
    grid_y = np.arange(H, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)          # both (H, W)

    map_x = (grid_x + flow[..., 0]).astype(np.float32)    # (H, W)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)    # (H, W)

    # cv2.remap expects float32 maps
    if frame.dtype == np.uint8:
        warped = cv2.remap(frame, map_x, map_y, interpolation, borderMode=border_mode)
    else:
        # float frames — remap, then clamp to original range
        src_min, src_max = frame.min(), frame.max()
        warped = cv2.remap(
            frame.astype(np.float32), map_x, map_y, interpolation, borderMode=border_mode,
        )
        warped = np.clip(warped, src_min, src_max)

    return warped


def consistency_score(
    fwd_flow: np.ndarray,
    bwd_flow: np.ndarray,
    alpha:    float = 0.01,
    beta:     float = 0.5,
) -> np.ndarray:
    """
    Compute per-pixel flow cycle-consistency score in [0, 1].

    Algorithm:
      1. Warp the backward flow field using the forward flow:
             warped_bwd[y, x] = bwd_flow at position (x + fwd_dx, y + fwd_dy)
      2. Compute the cycle error vector:
             cycle_err = fwd_flow + warped_bwd
         A consistent pixel satisfies fwd + warp(bwd) ≈ 0 — going forward
         and then backward returns you to the same place.
      3. Normalise the error relative to the flow magnitudes and compute a
         soft score using an exponential decay:
             score = exp( -||cycle_err|| / (alpha * (||fwd|| + ||warped_bwd||) + beta) )

    Parameters:
        fwd_flow: (H, W, 2) forward flow (frame_t → frame_{t+1})
        bwd_flow: (H, W, 2) backward flow (frame_{t+1} → frame_t)
        alpha:    relative threshold — pixels where the cycle error is less
                  than alpha * (sum of magnitudes) score near 1.0.
                  Lower alpha = stricter consistency requirement.
        beta:     absolute floor on the denominator — prevents division by
                  near-zero when there is very little motion.

    Returns:
        score: float32 array (H, W) in [0, 1]
               1 = perfectly consistent (static background)
               0 = completely inconsistent (occlusion or fast motion)

    Usage in flicker_suppressor.py (Tool 2):
        score = consistency_score(fwd_flow, bwd_flow)
        # High score pixels come from new diffusion output (correct style)
        # Low score pixels blend toward warped-previous frame (no flicker)
        output = blend_frames(new_styled, warped_prev, alpha=score)
    """
    # Step 1: warp backward flow using forward flow
    warped_bwd = warp_frame(bwd_flow, fwd_flow, interpolation=cv2.INTER_LINEAR)

    # Step 2: cycle error
    cycle_err   = fwd_flow + warped_bwd                          # (H, W, 2)
    err_mag     = np.linalg.norm(cycle_err, axis=-1)             # (H, W)

    # Step 3: normalised exponential score
    fwd_mag     = np.linalg.norm(fwd_flow,   axis=-1)            # (H, W)
    bwd_mag     = np.linalg.norm(warped_bwd, axis=-1)            # (H, W)
    denominator = alpha * (fwd_mag + bwd_mag) + beta

    score = np.exp(-err_mag / denominator).astype(np.float32)    # (H, W)

    return score


def blend_frames(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    alpha:   np.ndarray | float,
) -> np.ndarray:
    """
    Per-pixel alpha blend:  output = alpha * frame_a + (1 - alpha) * frame_b

    Args:
        frame_a: uint8 numpy array (H, W, C) — "foreground" (new diffusion output)
        frame_b: uint8 numpy array (H, W, C) — "background" (warped previous frame)
        alpha:   float32 array (H, W) in [0, 1], or a scalar float.
                 When alpha = 1 everywhere → pure frame_a.
                 When alpha = 0 everywhere → pure frame_b.

    Returns:
        Blended frame as uint8 numpy array (H, W, C).

    Usage in flicker_suppressor.py:
        # High consistency pixels → alpha ≈ 1 → take new styled frame
        # Low  consistency pixels → alpha ≈ 0 → take warped prev (temporally stable)
        output = blend_frames(new_styled, warped_prev, alpha=consistency_score(...))
    """
    a = frame_a.astype(np.float32)
    b = frame_b.astype(np.float32)

    if isinstance(alpha, np.ndarray):
        # Broadcast (H, W) → (H, W, 1) so it multiplies all channels
        w = alpha[..., np.newaxis]
    else:
        w = float(alpha)

    blended = w * a + (1.0 - w) * b
    return np.clip(blended, 0, 255).astype(np.uint8)


# ── Additional utilities used by mask_propagator and flicker_suppressor ────────

def soften_mask(mask: np.ndarray, kernel_size: int = 5, sigma: float = 2.0) -> np.ndarray:
    """
    Apply Gaussian blur to a binary or float mask to create soft edges.

    Hard mask edges cause visible "cut-out" artefacts in inpainting.
    Softening blends the mask into the surrounding image content so
    ProPainter and E2FGVI produce natural-looking boundaries.

    Args:
        mask:        float32 array (H, W) in [0, 1]
        kernel_size: Gaussian kernel size (must be odd)
        sigma:       Standard deviation of the Gaussian

    Returns:
        Softened mask as float32 array (H, W) in [0, 1].
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
    return blurred.astype(np.float32)


def dilate_mask(mask: np.ndarray, dilation_px: int) -> np.ndarray:
    """
    Morphologically dilate a binary mask by `dilation_px` pixels.

    Dilation expands the masked region outward, which:
      • Catches object boundaries that segmentation slightly under-estimates
      • Gives the inpainting model more context pixels to fill from

    Args:
        mask:         uint8 binary array (H, W) — 255 = masked, 0 = background
        dilation_px:  Radius of the dilation in pixels (0 = no dilation)

    Returns:
        Dilated mask as uint8 array (H, W).
    """
    if dilation_px == 0:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilation_px + 1, 2 * dilation_px + 1),
    )
    return cv2.dilate(mask, kernel, iterations=1)


def flow_magnitude_map(flow: np.ndarray) -> np.ndarray:
    """
    Compute a per-pixel motion magnitude map from a flow field.

    Useful for visualising where motion is occurring and for debugging
    flow-guided operations.  High values = fast motion, 0 = static.

    Args:
        flow: float32 array (H, W, 2)

    Returns:
        float32 array (H, W) — Euclidean magnitude at each pixel.
    """
    return np.linalg.norm(flow, axis=-1).astype(np.float32)


def threshold_by_motion(
    flow:      np.ndarray,
    threshold: float = 1.0,
) -> np.ndarray:
    """
    Return a binary mask of pixels with motion magnitude above `threshold`.

    Used to restrict style transfer processing to moving regions only,
    leaving static background unchanged (saves inference time on large batches).

    Args:
        flow:      float32 (H, W, 2) flow field
        threshold: pixels with magnitude < threshold are considered static

    Returns:
        uint8 array (H, W) — 255 where motion > threshold, 0 elsewhere.
    """
    mag    = flow_magnitude_map(flow)
    motion = (mag > threshold).astype(np.uint8) * 255
    return motion
