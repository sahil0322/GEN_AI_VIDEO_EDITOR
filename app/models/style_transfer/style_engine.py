# ==============================================================================
# app/models/style_transfer/style_engine.py
#
# Applies a text-driven visual style to a batch of video frames.
#
# Two local backends + one cloud fallback:
#
#   LocalSVDBackend          (backbone="svd")
#     Stable Video Diffusion — "stabilityai/stable-video-diffusion-img2vid-xt"
#     Workflow:
#       1. Apply SD img2img to frame 0 → styled keyframe
#       2. Feed styled keyframe into SVD → generates T temporally-smooth frames
#     Advantage:  SVD's motion prior produces the most temporally-consistent
#                 output; flickering is minimal even without the suppressor.
#     Limitation: text prompt influence is indirect (applied to keyframe only).
#     VRAM: ~12 GB
#
#   LocalControlNetBackend   (backbone="controlnet")
#     ControlNet (Canny) + SD 1.5 img2img — per-frame inference.
#     Workflow:
#       For each frame t:
#         1. Extract Canny edges from source frame  →  structural guide
#         2. Use flow-guided init image (from flow_guidance.py)  →  temporal guide
#         3. Run ControlNet img2img(prompt, canny, init)  →  styled frame
#     Advantage:  strong text prompt control; lower VRAM than SVD.
#     Limitation: frames are styled independently → needs flow_guidance +
#                 flicker_suppressor for temporal smoothness.
#     VRAM: ~4 GB
#
#   ReplicateBackend          (use_replicate=True / no local GPU)
#     Calls the Replicate API.  No local GPU needed.
#
# Interface contract (same for all backends):
#   Input:  source_frames  List[np.ndarray]  — uint8 BGR  (H, W, 3)
#           prompt         str               — style description
#           strength       float             — 0.1 – 1.0
#           steps          int               — diffusion steps
#           init_frames    Optional[List]    — flow-guided init from flow_guidance.py
#   Output: List[np.ndarray]  — uint8 BGR styled frames
# ==============================================================================

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

async def transfer(
    source_frames: list[np.ndarray],
    prompt:        str,
    strength:      float = 0.75,
    steps:         int   = 25,
    backbone:      str   = "svd",
    use_replicate: bool  = False,
    init_frames:   Optional[list[Optional[np.ndarray]]] = None,
) -> list[np.ndarray]:
    """
    Apply prompt-driven style transfer to every frame in the batch.

    Args:
        source_frames: Original BGR frames from the video
        prompt:        Style description, e.g. "a snowy winter landscape, cinematic"
        strength:      How strongly the style overrides the source (0.1 = subtle, 1.0 = total)
        steps:         Diffusion inference steps (more = better quality, slower)
        backbone:      "svd" | "controlnet"
        use_replicate: Route to Replicate cloud API
        init_frames:   Optional per-frame init images from flow_guidance.py.
                       When provided, each frame's diffusion starts from the
                       flow-warped previous styled frame rather than the raw
                       source frame, dramatically reducing inter-frame jumps.

    Returns:
        List of styled uint8 BGR frames, same length as source_frames.
    """
    if not source_frames:
        return []

    backend = _select_backend(backbone, use_replicate)
    logger.info(
        "style_engine: %d frames  prompt='%s'  strength=%.2f  "
        "steps=%d  backend=%s  flow_guided=%s",
        len(source_frames), prompt[:60], strength, steps,
        backend.__class__.__name__, init_frames is not None,
    )
    return await backend.run(source_frames, prompt, strength, steps, init_frames)


# ── Backend selection ──────────────────────────────────────────────────────────

def _select_backend(backbone: str, use_replicate: bool) -> "_StyleBackend":
    if use_replicate or not torch.cuda.is_available():
        if not settings.replicate_api_token:
            raise RuntimeError(
                "No local GPU detected and REPLICATE_API_TOKEN is not set."
            )
        return ReplicateStyleBackend(backbone=backbone)

    if backbone == "controlnet":
        return LocalControlNetBackend()

    return LocalSVDBackend()


# ── Abstract base ──────────────────────────────────────────────────────────────

class _StyleBackend(ABC):
    @abstractmethod
    async def run(
        self,
        frames:      list[np.ndarray],
        prompt:      str,
        strength:    float,
        steps:       int,
        init_frames: Optional[list],
    ) -> list[np.ndarray]:
        ...


# ══════════════════════════════════════════════════════════════════════════════
# Backend 1 — Stable Video Diffusion
# ══════════════════════════════════════════════════════════════════════════════

class LocalSVDBackend(_StyleBackend):
    """
    SVD workflow:
      Step A — Style the first frame with SD img2img (text-prompt controlled)
      Step B — Feed the styled keyframe into SVD to generate T smooth frames

    SVD's video-diffusion prior enforces temporal smoothness across all T frames
    simultaneously — it's the strongest source of temporal consistency in the
    entire pipeline, stronger than RAFT warping alone.

    Limitation: SVD generates a fixed-length clip (typically 14 or 25 frames)
    from one keyframe.  For batches larger than that we tile: style a new
    keyframe every SVD_CLIP_LEN frames and cross-fade at boundaries.
    """

    _sd_pipe  = None   # SD img2img for keyframe styling
    _svd_pipe = None   # SVD for temporal propagation

    SVD_CLIP_LEN  = 14   # frames generated per SVD call (model default)
    SVD_MODEL_ID  = "stabilityai/stable-video-diffusion-img2vid-xt"
    SD_MODEL_ID   = "runwayml/stable-diffusion-v1-5"

    async def run(self, frames, prompt, strength, steps, init_frames):
        return await asyncio.to_thread(
            self._run_sync, frames, prompt, strength, steps, init_frames
        )

    def _run_sync(self, frames, prompt, strength, steps, init_frames):
        self._ensure_loaded()
        from PIL import Image

        styled_all: list[np.ndarray] = []
        n = len(frames)
        clip_len = self.SVD_CLIP_LEN

        for start in range(0, n, clip_len):
            clip_frames  = frames[start : start + clip_len]
            clip_inits   = (init_frames or [None] * n)[start : start + clip_len]

            # ── A: Style keyframe (frame 0 of this clip) with SD img2img ──────
            key_frame  = clip_frames[0]
            key_init   = clip_inits[0]

            # Use flow-guided init if available, else use source frame
            init_pil = _bgr_to_pil(key_init if key_init is not None else key_frame)
            src_pil  = _bgr_to_pil(key_frame)

            with torch.no_grad():
                styled_key_pil = self._sd_pipe(
                    prompt            = prompt,
                    image             = init_pil,
                    strength          = min(strength, 0.9),  # cap so structure is kept
                    num_inference_steps = steps,
                    guidance_scale    = 8.0,
                ).images[0]

            # ── B: SVD propagates the styled keyframe across the clip ─────────
            with torch.no_grad():
                svd_frames = self._svd_pipe(
                    image             = styled_key_pil,
                    num_frames        = len(clip_frames),
                    num_inference_steps = max(steps // 2, 10),  # SVD needs fewer steps
                    decode_chunk_size = 4,
                    motion_bucket_id  = 127,   # controls motion amplitude (0-255)
                    noise_aug_strength = 0.02,
                ).frames[0]   # list[PIL.Image]

            # ── Convert PIL frames back to BGR numpy ──────────────────────────
            for pil_frame in svd_frames[: len(clip_frames)]:
                styled_all.append(_pil_to_bgr(pil_frame))

        return styled_all[:n]

    def _ensure_loaded(self) -> None:
        if self.__class__._svd_pipe is not None:
            return

        try:
            from diffusers import (  # type: ignore
                StableDiffusionImg2ImgPipeline,
                StableVideoDiffusionPipeline,
            )
        except ImportError as exc:
            raise ImportError(
                "diffusers not found or too old.\n"
                "  pip install diffusers==0.27.2 transformers accelerate\n"
                f"Original error: {exc}"
            ) from exc

        device     = torch.device("cuda")
        dtype      = torch.float16

        logger.info("Loading SD img2img for SVD keyframe styling…")
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.SD_MODEL_ID,
            torch_dtype = dtype,
            variant     = "fp16",
            safety_checker = None,
        ).to(device)
        sd_pipe.enable_attention_slicing()

        logger.info("Loading Stable Video Diffusion (this may take a while)…")
        svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.SVD_MODEL_ID,
            torch_dtype  = dtype,
            variant      = "fp16",
        ).to(device)
        svd_pipe.enable_model_cpu_offload()   # reduce peak VRAM by offloading layers

        self.__class__._sd_pipe  = sd_pipe
        self.__class__._svd_pipe = svd_pipe
        logger.info("SVD backend loaded  device=%s", device)


# ══════════════════════════════════════════════════════════════════════════════
# Backend 2 — ControlNet + SD 1.5 img2img
# ══════════════════════════════════════════════════════════════════════════════

class LocalControlNetBackend(_StyleBackend):
    """
    Per-frame style transfer using ControlNet (Canny) + SD 1.5 img2img.

    ControlNet conditioning:
      Canny edges are extracted from the SOURCE frame (not the styled one)
      so the structural outline of objects is preserved even at high strength.
      This prevents the "melted face" artefact where high-strength img2img
      destroys fine structural detail.

    Flow guidance integration:
      When init_frames[t] is provided (from flow_guidance.py), it's used
      as the img2img init image instead of the raw source frame.  This means
      the diffusion starts from the flow-warped previous styled frame —
      the model only needs to make small adjustments rather than re-styling
      from scratch, which is where the temporal smoothness comes from.
    """

    _pipe = None
    _CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"
    _SD_MODEL         = "runwayml/stable-diffusion-v1-5"

    async def run(self, frames, prompt, strength, steps, init_frames):
        return await asyncio.to_thread(
            self._run_sync, frames, prompt, strength, steps, init_frames
        )

    def _run_sync(self, frames, prompt, strength, steps, init_frames):
        self._ensure_loaded()
        from PIL import Image

        styled = []
        n = len(frames)

        for t, frame in enumerate(frames):
            orig_h, orig_w = frame.shape[:2]
            if orig_w > 768:
                scale = 768 / orig_w
                new_w, new_h = 768, int(orig_h * scale)
                # Ensure dimensions are divisible by 8 for Stable Diffusion
                new_w = (new_w // 8) * 8
                new_h = (new_h // 8) * 8
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # ── Canny edges from source (structural conditioning) ─────────────
            canny = _extract_canny(frame)
            canny_pil = Image.fromarray(canny).convert("RGB")

            # ── Init image: flow-guided or raw source ─────────────────────────
            # flow_guidance.py provides init_frames[t] = warp(prev_styled, fwd_flow)
            # When available, diffusion starts from a temporally-aware prior.
            if init_frames and init_frames[t] is not None:
                init_pil = _bgr_to_pil(init_frames[t])
            else:
                init_pil = _bgr_to_pil(frame)

            with torch.no_grad():
                result = self._pipe(
                    prompt              = prompt,
                    image               = init_pil,       # img2img init
                    control_image       = canny_pil,      # ControlNet conditioning
                    strength            = strength,
                    num_inference_steps = steps,
                    guidance_scale      = 7.5,
                    controlnet_conditioning_scale = 0.8,  # weight of canny guide
                ).images[0]

            styled_bgr = _pil_to_bgr(result)

            # ── 4-PIXEL FIX: Force SD's output back to original frame dimensions ──
            orig_h, orig_w = frame.shape[:2]
            if styled_bgr.shape[:2] != (orig_h, orig_w):
                styled_bgr = cv2.resize(styled_bgr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

            styled.append(styled_bgr)
            logger.debug("ControlNet frame %d/%d done", t + 1, n)

        return styled

    def _ensure_loaded(self) -> None:
        if self.__class__._pipe is not None:
            return

        try:
            from diffusers import (  # type: ignore
                ControlNetModel,
                StableDiffusionControlNetImg2ImgPipeline,
                UniPCMultistepScheduler,
            )
        except ImportError as exc:
            raise ImportError(
                "diffusers not found.\n"
                "  pip install diffusers==0.27.2 transformers accelerate\n"
                f"Original error: {exc}"
            ) from exc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype  = torch.float16 if device.type == "cuda" else torch.float32

        logger.info("Loading ControlNet…")
        controlnet = ControlNetModel.from_pretrained(
            self._CONTROLNET_MODEL, torch_dtype=dtype,
        )

        logger.info("Loading SD 1.5 ControlNet img2img pipeline…")
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self._SD_MODEL,
            controlnet     = controlnet,
            torch_dtype    = dtype,
            safety_checker = None,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()

        self.__class__._pipe = pipe
        logger.info("ControlNet backend loaded  device=%s", device)


# ══════════════════════════════════════════════════════════════════════════════
# Backend 3 — Replicate cloud API
# ══════════════════════════════════════════════════════════════════════════════

class ReplicateStyleBackend(_StyleBackend):
    """
    Calls the Replicate-hosted ControlNet style transfer model.
    Used when no local GPU is available or use_replicate=True.
    """

    _MODEL_ID = "jagilley/controlnet-canny:aff48af9c68d162388d230a2ab003f68d2638d88"

    def __init__(self, backbone: str = "controlnet") -> None:
        self._backbone = backbone

    async def run(self, frames, prompt, strength, steps, init_frames):
        return await asyncio.to_thread(
            self._run_sync, frames, prompt, strength, steps, init_frames
        )

    def _run_sync(self, frames, prompt, strength, steps, init_frames):
        import os, base64, urllib.request, tempfile
        try:
            import replicate  # type: ignore
        except ImportError as exc:
            raise ImportError("pip install replicate") from exc

        os.environ["REPLICATE_API_TOKEN"] = settings.replicate_api_token

        styled = []
        for t, frame in enumerate(frames):
            init_f  = (init_frames or [None] * len(frames))[t]
            img_b64 = _frame_to_b64(init_f if init_f is not None else frame)
            canny   = _extract_canny(frame)
            canny_b64 = _frame_to_b64(cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR))

            output_url = replicate.run(
                self._MODEL_ID,
                input={
                    "image":  img_b64,
                    "prompt": prompt,
                    "a_prompt": "best quality, extremely detailed",
                    "n_prompt": "longbody, lowres, bad anatomy, extra digit, fewer digits",
                    "num_samples":   "1",
                    "image_resolution": "512",
                    "low_threshold": 100,
                    "high_threshold": 200,
                    "ddim_steps":   steps,
                    "scale":        7.5,
                    "eta":          0.0,
                    "strength":     strength,
                },
            )
            url = output_url[0] if isinstance(output_url, list) else output_url
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                urllib.request.urlretrieve(url, tmp.name)
                result = cv2.imread(tmp.name)
            styled.append(result)

        return styled


# ── Image conversion helpers ───────────────────────────────────────────────────

def _bgr_to_pil(frame: np.ndarray):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def _pil_to_bgr(pil_img) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def _extract_canny(frame: np.ndarray, lo: int = 100, hi: int = 200) -> np.ndarray:
    """Extract Canny edges as a single-channel uint8 array."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, lo, hi)

def _frame_to_b64(frame: np.ndarray) -> str:
    import base64
    _, buf = cv2.imencode(".png", frame)
    return "data:image/png;base64," + base64.b64encode(buf).decode()
