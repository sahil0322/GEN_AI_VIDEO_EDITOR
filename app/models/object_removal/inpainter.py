# ==============================================================================
# app/models/object_removal/inpainter.py
#
# Fills the masked regions with temporally-consistent background content.
#
# Strategy pattern — three backends, selected at runtime:
#
#   LocalProPainterBackend   (default)
#     Runs ProPainter locally.  Best quality.  Requires ~6 GB VRAM.
#     ProPainter is a recurrent flow-guided inpainting model designed
#     specifically for video — it uses optical flow to propagate known
#     background pixels forward in time before hallucinating new content.
#
#   LocalE2FGVIBackend       (fallback when VRAM < 6 GB)
#     Runs E2FGVI locally.  Good quality, ~4 GB VRAM, faster.
#     End-to-end flow-guided video inpainting — similar approach to
#     ProPainter but lighter.
#
#   ReplicateBackend          (cloud GPU fallback)
#     Calls the Replicate API when no local GPU is available or when
#     use_replicate=True.  Sends frames as base64 PNGs, receives output
#     as a URL, downloads and decodes results.
#
# Both local backends share the same input/output contract:
#   Input:  List of BGR uint8 numpy frames  +  List of uint8 masks (0/255)
#   Output: List of BGR uint8 numpy frames  (inpainted, same shape)
#
# Temporal consistency in ProPainter:
#   ProPainter internally computes RAFT flow between all frame pairs in the
#   batch and uses it to warp known background pixels into the masked region
#   before the transformer fills remaining holes.  This is why it produces
#   much fewer "smearing" artefacts than single-frame inpainters.
#   Our mask_propagator.py pre-computed the same flow fields — ProPainter
#   can optionally accept them to avoid recomputing (Step 8 wires this).
# ==============================================================================

from __future__ import annotations

import base64
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

async def inpaint(
    frames:        list[np.ndarray],
    masks:         list[np.ndarray],
    model_type:    str  = "propainter",
    use_replicate: bool = False,
) -> list[np.ndarray]:
    """
    Fill masked regions in each frame with temporally-consistent background.

    Args:
        frames:        List of uint8 BGR numpy arrays (H, W, 3)
        masks:         List of uint8 (H, W) arrays — 255 = inpaint here, 0 = keep
        model_type:    "propainter" | "e2fgvi"  (ignored when use_replicate=True)
        use_replicate: True → use Replicate cloud API regardless of local GPU

    Returns:
        List of inpainted uint8 BGR numpy arrays, same length as `frames`.

    Raises:
        ValueError: frames and masks have different lengths
        RuntimeError: backend inference failed
    """
    if len(frames) != len(masks):
        raise ValueError(
            f"frames ({len(frames)}) and masks ({len(masks)}) must have the same length"
        )
    if not frames:
        return []

    backend = _select_backend(model_type, use_replicate)
    logger.info(
        "inpainter: %d frames  model=%s  backend=%s",
        len(frames), model_type, backend.__class__.__name__,
    )
    return await backend.run(frames, masks)


# ── Backend selection ──────────────────────────────────────────────────────────

def _select_backend(model_type: str, use_replicate: bool) -> "_InpaintBackend":
    if use_replicate or not torch.cuda.is_available():
        if not settings.replicate_api_token:
            raise RuntimeError(
                "No local GPU detected and REPLICATE_API_TOKEN is not set.\n"
                "Either connect a GPU or set REPLICATE_API_TOKEN in your .env file."
            )
        return ReplicateBackend(model_type=model_type)

    if model_type == "e2fgvi":
        return LocalE2FGVIBackend()

    return LocalProPainterBackend()


# ── Abstract base ──────────────────────────────────────────────────────────────

class _InpaintBackend(ABC):

    @abstractmethod
    async def run(
        self,
        frames: list[np.ndarray],
        masks:  list[np.ndarray],
    ) -> list[np.ndarray]:
        ...


# ── Backend 1: Local ProPainter ────────────────────────────────────────────────

class LocalProPainterBackend(_InpaintBackend):
    """
    Runs ProPainter (https://github.com/sczhou/ProPainter) locally.

    ProPainter expects:
      • frames as a float32 tensor  [T, 3, H, W]  in [0, 1]  (RGB)
      • masks  as a float32 tensor  [T, 1, H, W]  in [0, 1]
      • image size must be divisible by 8 (we pad if needed)
    """

    _model = None   # lazy-loaded

    async def run(
        self,
        frames: list[np.ndarray],
        masks:  list[np.ndarray],
    ) -> list[np.ndarray]:
        import asyncio
        return await asyncio.to_thread(self._run_sync, frames, masks)

    def _run_sync(
        self,
        frames: list[np.ndarray],
        masks:  list[np.ndarray],
    ) -> list[np.ndarray]:
        self._ensure_loaded()
        import cv2

        from app.models.object_removal.mask_propagator import (
            frames_to_tensor_batch,
            masks_to_tensor_batch,
        )

        device = next(self._model.parameters()).device

        # ── AUTO-SCALER (Saves your 8GB GPU) ──────────────────────────────────
        orig_h, orig_w = frames[0].shape[:2]
        # Shrink to 480p max to prevent OOM memory crashes
        scale = min(1.0, 480.0 / orig_h)
        work_h, work_w = int(orig_h * scale), int(orig_w * scale)
        
        if scale < 1.0:
            frames = [cv2.resize(f, (work_w, work_h), interpolation=cv2.INTER_AREA) for f in frames]
            masks  = [cv2.resize(m, (work_w, work_h), interpolation=cv2.INTER_NEAREST) for m in masks]

        # ── Convert inputs ────────────────────────────────────────────────────
        frames_t = frames_to_tensor_batch(frames).unsqueeze(0).to(device)
        masks_t  = masks_to_tensor_batch(masks).unsqueeze(0).to(device)

        _, T, C, H, W = frames_t.shape

        # Force exact padding to multiples of 8 (ProPainter requirement)
        import math
        pad_h = math.ceil(H / 8) * 8 - H
        pad_w = math.ceil(W / 8) * 8 - W
        if pad_h > 0 or pad_w > 0:
            import torch.nn.functional as F
            frames_t = F.pad(frames_t, (0, pad_w, 0, pad_h))
            masks_t  = F.pad(masks_t,  (0, pad_w, 0, pad_h))

        b, t, c, padded_h, padded_w = frames_t.shape

        # ── Create Fake Flow Tensors ──────────────────────────────────────────
        zero_flow_f = torch.zeros(b * (t - 1), 2, padded_h, padded_w, device=device)
        zero_flow_b = torch.zeros(b * (t - 1), 2, padded_h, padded_w, device=device)
        completed_flows = (zero_flow_f, zero_flow_b)

        # ── VRAM Cleanup ──────────────────────────────────────────────────────
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Inference (Half-Precision Mode) ───────────────────────────────────
       # ── Inference (Half-Precision Mode) ───────────────────────────────────
        with torch.no_grad(), torch.autocast("cuda"):
            output_raw = self._model(frames_t, completed_flows, masks_t, masks_t, min(t, 10))
            
            # Bulletproof catch: handles both single tensors and tuples
            if isinstance(output_raw, tuple):
                output_t = output_raw[0]
            else:
                output_t = output_raw

        # ── Crop padding and convert back to numpy ───────────────────────────
        output_t = output_t.squeeze(0)
        output_t = output_t[:, :, :H, :W]
        
        output_frames = _tensor_batch_to_bgr_frames(output_t)
        
        # ── SCALE BACK UP ─────────────────────────────────────────────────────
        if scale < 1.0:
            output_frames = [cv2.resize(f, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC) for f in output_frames]
            
        return output_frames

    def _ensure_loaded(self) -> None:
        if self.__class__._model is not None:
            return

        try:
            import sys
            import os
            
            # Point Python directly to the ProPainter directory
            propainter_path = os.path.abspath(os.path.join(os.getcwd(), "ProPainter"))
            if propainter_path not in sys.path:
                sys.path.append(propainter_path)
                
            # Now that it can see inside the folder, import the generator
            try:
                from model.propainter import InpaintGenerator  # type: ignore
            except ModuleNotFoundError:
                from propainter.model.propainter import InpaintGenerator  # type: ignore
                
        except ImportError as exc:
            raise ImportError(
                "ProPainter is not installed.\n"
                "  git clone https://github.com/sczhou/ProPainter.git\n"
                "  pip install -e ProPainter/\n"
                f"Original error: {exc}"
            ) from exc

        ckpt_path = Path(settings.propainter_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"ProPainter checkpoint not found at {ckpt_path}.\n"
                "Download from: https://github.com/sczhou/ProPainter/releases"
            )

        device = torch.device("cuda")
        model = InpaintGenerator(model_path="weights/ProPainter.pth")
        state  = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(state.get("netG", state), strict=False)
        model.to(device)
        model.eval()

        self.__class__._model = model
        logger.info("ProPainter loaded  device=%s", device)


# ── Backend 2: Local E2FGVI ───────────────────────────────────────────────────

class LocalE2FGVIBackend(_InpaintBackend):
    """
    Runs E2FGVI (https://github.com/MCG-NKU/E2FGVI) locally.
    Lower VRAM requirement (~4 GB) and faster than ProPainter.
    """

    _model = None

    async def run(
        self,
        frames: list[np.ndarray],
        masks:  list[np.ndarray],
    ) -> list[np.ndarray]:
        import asyncio
        return await asyncio.to_thread(self._run_sync, frames, masks)

    def _run_sync(
        self,
        frames: list[np.ndarray],
        masks:  list[np.ndarray],
    ) -> list[np.ndarray]:
        self._ensure_loaded()

        from app.models.object_removal.mask_propagator import (
            frames_to_tensor_batch,
            masks_to_tensor_batch,
        )

        device = next(self._model.parameters()).device

        # E2FGVI requires [1, T, 3, H, W]
        frames_t = frames_to_tensor_batch(frames).unsqueeze(0).to(device)
        masks_t  = masks_to_tensor_batch(masks).unsqueeze(0).to(device)

        _, T, C, H, W = frames_t.shape

        # Pad to multiples of 8 to prevent math crashing
        import math
        pad_h = math.ceil(H / 8) * 8 - H
        pad_w = math.ceil(W / 8) * 8 - W
        if pad_h > 0 or pad_w > 0:
            import torch.nn.functional as F
            frames_t = F.pad(frames_t, (0, pad_w, 0, pad_h))
            masks_t  = F.pad(masks_t,  (0, pad_w, 0, pad_h))

        with torch.no_grad():
            output_t, _ = self._model(frames_t, masks_t)

        # Crop padding and remove the batch dimension
        output_t = output_t.squeeze(0)   
        output_t = output_t[:, :, :H, :W] 
        return _tensor_batch_to_bgr_frames(output_t)

    def _ensure_loaded(self) -> None:
        if self.__class__._model is not None:
            return

        import sys
        import os
        
        e2fgvi_path = os.path.abspath(os.path.join(os.getcwd(), "E2FGVI"))
        
        if e2fgvi_path not in sys.path:
            sys.path.insert(0, e2fgvi_path)

        try:
            # CORRECT E2FGVI IMPORT PATH
            from model.e2fgvi_hq import InpaintGenerator
        except ImportError as exc:
            raise ImportError(
                "E2FGVI is not installed.\n"
                "  git clone https://github.com/MCG-NKU/E2FGVI.git\n"
                f"Original error: {exc}"
            ) from exc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = InpaintGenerator()

        ckpt = Path("weights/E2FGVI-HQ-CVPR22.pth")
        if not ckpt.exists():
            raise FileNotFoundError("E2FGVI checkpoint not found at weights/E2FGVI-HQ-CVPR22.pth")

        state = torch.load(str(ckpt), map_location=device)
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()

        self.__class__._model = model
        logger.info("E2FGVI loaded  device=%s", device)


# ── Backend 3: Replicate cloud API ────────────────────────────────────────────

class ReplicateBackend(_InpaintBackend):
    """
    Offloads inference to Replicate.com cloud GPUs.
    """

    _REPLICATE_MODEL_ID = "sczhou/propainter:3197cf2f"

    def __init__(self, model_type: str = "propainter") -> None:
        self._model_type = model_type

    async def run(
        self,
        frames: list[np.ndarray],
        masks:  list[np.ndarray],
    ) -> list[np.ndarray]:
        import asyncio
        return await asyncio.to_thread(self._run_sync, frames, masks)

    def _run_sync(
        self,
        frames: list[np.ndarray],
        masks:  list[np.ndarray],
    ) -> list[np.ndarray]:
        try:
            import replicate  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "replicate package not found.  pip install replicate\n"
                f"Original error: {exc}"
            ) from exc

        api_token = settings.replicate_api_token
        if not api_token:
            raise RuntimeError("REPLICATE_API_TOKEN is not set in .env")

        os.environ["REPLICATE_API_TOKEN"] = api_token

        with tempfile.TemporaryDirectory(prefix="flowedit_replicate_") as tmpdir:
            tmp = Path(tmpdir)

            for i, (frame, mask) in enumerate(zip(frames, masks)):
                cv2.imwrite(str(tmp / f"frame_{i:04d}.png"), frame)
                cv2.imwrite(str(tmp / f"mask_{i:04d}.png"),  mask)

            frame_files = sorted(tmp.glob("frame_*.png"))
            mask_files  = sorted(tmp.glob("mask_*.png"))

            frames_b64 = [_file_to_data_uri(f) for f in frame_files]
            masks_b64  = [_file_to_data_uri(f) for f in mask_files]

            logger.info(
                "Calling Replicate %s with %d frames",
                self._REPLICATE_MODEL_ID, len(frames),
            )

            output = replicate.run(
                self._REPLICATE_MODEL_ID,
                input={
                    "frames": frames_b64,
                    "masks":  masks_b64,
                },
            )

            output_url = output if isinstance(output, str) else output[0]
            return self._download_and_decode(output_url, len(frames), tmpdir)

    def _download_and_decode(
        self,
        url:       str,
        n_frames:  int,
        tmpdir:    str,
    ) -> list[np.ndarray]:
        import urllib.request
        out_video = Path(tmpdir) / "replicate_output.mp4"
        urllib.request.urlretrieve(url, str(out_video))

        cap = cv2.VideoCapture(str(out_video))
        frames_out = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_out.append(frame)
        cap.release()

        if len(frames_out) != n_frames:
            logger.warning(
                "Replicate returned %d frames but expected %d",
                len(frames_out), n_frames,
            )
        return frames_out[:n_frames]


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _tensor_batch_to_bgr_frames(tensor: "torch.Tensor") -> list[np.ndarray]:
    """
    Convert a [T, 3, H, W] float32 tensor in [0, 1] (RGB) back to a list
    of uint8 BGR numpy frames.
    """
    frames = []
    for t in range(tensor.shape[0]):
        frame_rgb = (
            tensor[t]
            .permute(1, 2, 0)
            .clamp(0, 1)
            .mul(255)
            .byte()
            .cpu()
            .numpy()
        )
        frames.append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    return frames


def _file_to_data_uri(path: Path) -> str:
    """Read a file and return it as a base64 data URI for the Replicate API."""
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode("utf-8")
    ext  = path.suffix.lstrip(".")
    return f"data:image/{ext};base64,{data}"