# ==============================================================================
# app/models/object_removal/segmenter.py
#
# Generates a per-pixel binary segmentation mask for the target object.
#
# Two segmentation paths, selected by `detection_backend` from the frontend:
#
#   "grounding_dino"  (default)
#     Text prompt  →  GroundingDINO  →  bounding box(es)
#                  →  SAM predictor  →  binary mask
#     Best for: natural language descriptions ("the red car on the left")
#
#   "sam_bbox"
#     User-drawn bounding box (x0, y0, x1, y1 in pixel coords)
#                  →  SAM predictor  →  binary mask
#     Best for: precise spatial selection when text is ambiguous
#
# Both paths terminate at SAM, so the output mask format is identical
# regardless of which backend was used — mask_propagator.py doesn't need
# to know which path was taken.
#
# Lazy loading:
#   Both GroundingDINO and SAM are large models.  They are loaded on the
#   first call to segment() and cached in module-level singletons for the
#   lifetime of the server process.  Subsequent calls within the same job
#   (or different jobs) reuse the loaded weights.
# ==============================================================================

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Module-level singletons (lazy-loaded on first use) ─────────────────────────
_gdino_model  = None   # GroundingDINO
_sam_predictor = None  # SAM SamPredictor

# Detection thresholds for GroundingDINO
_BOX_THRESHOLD  = 0.35
_TEXT_THRESHOLD = 0.25


# ── Public API ─────────────────────────────────────────────────────────────────

def segment(
    frame:             np.ndarray,
    prompt:            Optional[str]   = None,
    bbox:              Optional[tuple] = None,
    detection_backend: str             = "grounding_dino",
    dilation_px:       int             = 8,
) -> np.ndarray:
    """
    Generate a binary segmentation mask for `frame`.

    Exactly one of `prompt` or `bbox` must be provided:
      • prompt: natural-language description of the object to remove
      • bbox:   (x0, y0, x1, y1) in pixel coordinates (from frontend canvas)

    Args:
        frame:             uint8 numpy array (H, W, 3)  — BGR (OpenCV default)
        prompt:            text description, e.g. "the person on the left"
        bbox:              pixel bounding box tuple (x0, y0, x1, y1)
        detection_backend: "grounding_dino" | "sam_bbox"
        dilation_px:       expand mask outward by this many pixels after SAM
                           (covers object edges that segmentation under-estimates)

    Returns:
        mask: uint8 numpy array (H, W)  — 255 = object/remove, 0 = background/keep

    Raises:
        ValueError:  neither prompt nor bbox provided
        RuntimeError: GroundingDINO finds no boxes matching the prompt
    """
    if prompt is None and bbox is None:
        raise ValueError("Either `prompt` or `bbox` must be provided to segment()")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # SAM / GDINO expect RGB

    # ── Path A: GroundingDINO → bounding box → SAM ────────────────────────────
    if detection_backend == "grounding_dino" and prompt:
        box_xyxy = _run_grounding_dino(frame_rgb, prompt)
    # ── Path B: user-supplied bounding box → SAM ──────────────────────────────
    elif bbox is not None:
        box_xyxy = np.array(bbox, dtype=np.float32)
    else:
        raise ValueError(
            f"detection_backend='{detection_backend}' requires a bbox tuple. "
            "Set detection_backend='grounding_dino' to use a text prompt."
        )

    # ── SAM: bounding box → per-pixel mask ────────────────────────────────────
    mask = _run_sam(frame_rgb, box_xyxy)

    # ── Post-process: dilate to cover boundary pixels ─────────────────────────
    if dilation_px > 0:
        from app.pipeline.temporal.flow_utils import dilate_mask
        mask = dilate_mask(mask, dilation_px)

    logger.debug(
        "segment(): backend=%s  mask_coverage=%.1f%%  dilation=%dpx",
        detection_backend,
        100.0 * mask.sum() / mask.size,
        dilation_px,
    )
    return mask


# ── GroundingDINO detection ────────────────────────────────────────────────────
def _run_grounding_dino(frame_rgb: np.ndarray, prompt: str) -> np.ndarray:
    """
    Run GroundingDINO to detect bounding boxes for `prompt` in `frame_rgb`.

    Returns the highest-confidence box as a (4,) float32 array [x0, y0, x1, y1]
    in absolute pixel coordinates.

    Raises RuntimeError if no box is found above _BOX_THRESHOLD.
    """
    _ensure_gdino_loaded()

    try:
        from groundingdino.util.inference import predict         # type: ignore
        import groundingdino.datasets.transforms as T
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "GroundingDINO is not installed.\n"
            "  pip install git+https://github.com/IDEA-Research/GroundingDINO.git\n"
            f"Original error: {exc}"
        ) from exc

    H, W = frame_rgb.shape[:2]

    # Convert cv2 RGB frame to PIL Image
    pil_img = Image.fromarray(frame_rgb)

    # Use GroundingDINO's exact expected transforms
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # transform returns (image_tensor, target). We only need the image_tensor.
    # Note: NO .unsqueeze(0) here! predict() expects shape [3, H, W]
    image_tensor, _ = transform(pil_img, None)

    with torch.no_grad():
        boxes_norm, logits, phrases = predict(
            model      = _gdino_model,
            image      = image_tensor,
            caption    = prompt.lower().strip(),
            box_threshold  = _BOX_THRESHOLD,
            text_threshold = _TEXT_THRESHOLD,
        )

    if len(boxes_norm) == 0:
        raise RuntimeError(
            f"GroundingDINO found no objects matching '{prompt}' "
            f"(box_threshold={_BOX_THRESHOLD}). "
            "Try a more specific description or switch to bounding-box mode."
        )

    # Pick the highest-confidence detection
    best_idx  = logits.argmax().item()
    box_norm  = boxes_norm[best_idx].numpy()          # [cx, cy, w, h] normalised

    # Convert cxcywh normalised → xyxy absolute pixels
    cx, cy, bw, bh = box_norm
    x0 = int((cx - bw / 2) * W)
    y0 = int((cy - bh / 2) * H)
    x1 = int((cx + bw / 2) * W)
    y1 = int((cy + bh / 2) * H)

    logger.info(
        "GroundingDINO: '%s' → box [%d,%d,%d,%d]  confidence=%.3f",
        phrases[best_idx], x0, y0, x1, y1, logits[best_idx].item(),
    )
    return np.array([x0, y0, x1, y1], dtype=np.float32)


# ── SAM segmentation ───────────────────────────────────────────────────────────

def _run_sam(frame_rgb: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    """
    Run SAM to produce a per-pixel mask given a bounding box.

    SAM outputs three candidate masks (multimask_output=True) ranked by
    predicted IoU score.  We pick the highest-scoring one.

    Returns:
        uint8 array (H, W) — 255 = object, 0 = background
    """
    _ensure_sam_loaded()

    _sam_predictor.set_image(frame_rgb)

    # SAM expects box as [x0, y0, x1, y1] float32
    box_input = box_xyxy.reshape(1, 4)

    masks, iou_preds, _ = _sam_predictor.predict(
        box              = box_input,
        multimask_output = True,       # let SAM propose 3 candidates
    )

    # Pick the mask with the highest predicted IoU
    best_mask = masks[np.argmax(iou_preds)]   # (H, W) bool

    result = (best_mask.astype(np.uint8)) * 255
    logger.debug(
        "SAM: best IoU=%.3f  mask_px=%d",
        iou_preds.max(), result.sum() // 255,
    )
    return result


# ── Lazy loaders ───────────────────────────────────────────────────────────────

def _ensure_gdino_loaded() -> None:
    global _gdino_model
    if _gdino_model is not None:
        return

    try:
        from groundingdino.util.inference import load_model  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "GroundingDINO not found. Install with:\n"
            "  pip install git+https://github.com/IDEA-Research/GroundingDINO.git\n"
            f"Original error: {exc}"
        ) from exc

    config  = settings.grounding_dino_config
    weights = settings.grounding_dino_weights

    if not Path(weights).exists():
        _download_gdino_weights(Path(weights))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _gdino_model = load_model(config, weights, device=device)
    _gdino_model.eval()
    logger.info("GroundingDINO loaded  device=%s", device)


def _ensure_sam_loaded() -> None:
    global _sam_predictor
    if _sam_predictor is not None:
        return

    try:
        from segment_anything import SamPredictor, sam_model_registry  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "segment-anything not found. Install with:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
            f"Original error: {exc}"
        ) from exc

    checkpoint = settings.sam_checkpoint
    model_type = settings.sam_model_type

    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {checkpoint}.\n"
            "Download it with:\n"
            "  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth "
            "-P weights/"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam    = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    _sam_predictor = SamPredictor(sam)
    logger.info("SAM loaded  model=%s  device=%s", model_type, device)


def _download_gdino_weights(dest: Path) -> None:
    """Auto-download GroundingDINO SwinT weights if missing."""
    import urllib.request
    url = (
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
        "v0.1.0-alpha/groundingdino_swint_ogc.pth"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading GroundingDINO weights → %s", dest)
    urllib.request.urlretrieve(url, str(dest))
    logger.info("GroundingDINO weights downloaded.")

def unload_models():
    """Forces the GPU to clear GroundingDINO and SAM to free up VRAM for RAFT/Inpainters."""
    global _gdino_model, _sam_predictor
    
    try:
        if _gdino_model is not None:
            del _gdino_model
            _gdino_model = None
    except NameError:
        pass  # Ignore if it hasn't been defined yet
        
    try:
        if _sam_predictor is not None:
            del _sam_predictor
            _sam_predictor = None
    except NameError:
        pass  # Ignore if it hasn't been defined yet
        
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Cleared GroundingDINO and SAM from GPU memory.")