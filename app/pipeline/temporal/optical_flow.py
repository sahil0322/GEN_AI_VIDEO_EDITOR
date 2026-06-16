# ==============================================================================
# app/pipeline/temporal/optical_flow.py
#
# RAFT optical flow engine — shared by both AI tools.
#
# What is optical flow?
#   A dense 2-D vector field where every pixel (x, y) carries a displacement
#   vector (dx, dy) describing where that pixel MOVED between two consecutive
#   frames.  Given frame_t and frame_{t+1}:
#     fwd_flow[y, x] = (dx, dy)  →  pixel at (x,y) in frame_t moved to
#                                    (x+dx, y+dy) in frame_{t+1}
#     bwd_flow[y, x] = (dx, dy)  →  pixel at (x,y) in frame_{t+1} came
#                                    from (x-dx, y-dy) in frame_t
#
# Why shared?
#   Both Tool 1 (mask_propagator.py) and Tool 2 (flow_guidance.py +
#   flicker_suppressor.py) need the same flow fields for the same frame pair.
#   A single RAFTFlowEngine instance created once in orchestrator._run() and
#   passed to both tools ensures:
#     • No duplicate GPU memory for two RAFT checkpoints
#     • Identical flow fields for mask warping AND style conditioning
#     • Consistent temporal behaviour when both tools are eventually chained
#
# Usage pattern in orchestrator._run() (Step 8 completes this):
#
#   flow_engine = RAFTFlowEngine(settings.raft_checkpoint)
#
#   # Tool 1
#   for frame_t, frame_t1 in zip(batch[:-1], batch[1:]):
#       fwd, bwd = await flow_engine.compute_flow_async(frame_t, frame_t1)
#       warped_mask = flow_utils.warp_frame(mask_t.astype("float32"), fwd)
#
#   # Tool 2
#   for frame_t, frame_t1 in zip(batch[:-1], batch[1:]):
#       fwd, bwd = await flow_engine.compute_flow_async(frame_t, frame_t1)
#       score    = flow_utils.consistency_score(fwd, bwd)
#       output   = flow_utils.blend_frames(new_styled, warped_prev, score)
#
# RAFT installation (run once):
#   git clone https://github.com/princeton-vl/RAFT.git
#   pip install -e RAFT/
#   # Checkpoint download is handled automatically by _maybe_download_weights()
# ==============================================================================

from __future__ import annotations

import asyncio
import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Public URL for the raft-things checkpoint (~20 MB).
# "things" = trained on FlyingThings3D — best general-purpose flow model.
_RAFT_CHECKPOINT_URL = (
    "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/raft-things.pth"
)
# RAFT expects images as uint8 [0, 255] tensors (not float [0, 1]).
_RAFT_ITERS = 20        # recurrent update iterations; 12 for speed, 20 for quality


class RAFTFlowEngine:
    """
    Lazy-loading wrapper around RAFT optical flow.

    The RAFT model is loaded from the checkpoint on the first call to
    compute_flow() and cached for the lifetime of the engine instance.
    Subsequent calls reuse the loaded model with no startup overhead.

    Thread safety:
        compute_flow() is synchronous and not thread-safe (PyTorch tensors
        are not shared across threads without care).  Always call it via
        compute_flow_async() which runs it in a dedicated thread executor,
        or call it directly from a single thread.
    """

    def __init__(self, checkpoint_path: str | Path) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._model: Optional[object]     = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "RAFTFlowEngine created — device: %s  checkpoint: %s",
            self._device, self._checkpoint_path,
        )

    # ── Public interface ───────────────────────────────────────────────────────

    def compute_flow(
        self,
        frame_t:  np.ndarray,
        frame_t1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute forward and backward optical flow between two consecutive frames.

        Args:
            frame_t:  First frame  — uint8 numpy array (H, W, 3) BGR or RGB
            frame_t1: Second frame — uint8 numpy array (H, W, 3) BGR or RGB

        Returns:
            fwd_flow: np.float32 array (H, W, 2) — displacement from t → t+1
            bwd_flow: np.float32 array (H, W, 2) — displacement from t+1 → t

        Note:
            flow[y, x, 0] = horizontal displacement (dx)
            flow[y, x, 1] = vertical   displacement (dy)
        """
        self._ensure_loaded()

        # Convert to RGB float tensors expected by RAFT
        t0  = _frame_to_tensor(frame_t,  self._device)   # [1, 3, H, W]
        t1  = _frame_to_tensor(frame_t1, self._device)   # [1, 3, H, W]

        # Pad to multiples of 8 (RAFT architecture requirement)
        padder      = _InputPadder(t0.shape)
        t0_p, t1_p  = padder.pad(t0, t1)

        with torch.no_grad():
            _, fwd_up = self._model(t0_p, t1_p, iters=_RAFT_ITERS, test_mode=True)
            _, bwd_up = self._model(t1_p, t0_p, iters=_RAFT_ITERS, test_mode=True)

        # Remove padding and move to CPU numpy
        fwd_up = padder.unpad(fwd_up)
        bwd_up = padder.unpad(bwd_up)

        fwd_flow = _flow_tensor_to_numpy(fwd_up)   # (H, W, 2)
        bwd_flow = _flow_tensor_to_numpy(bwd_up)   # (H, W, 2)

        return fwd_flow, bwd_flow

    async def compute_flow_async(
        self,
        frame_t:  np.ndarray,
        frame_t1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Async wrapper — runs compute_flow() in a thread executor so RAFT
        inference doesn't block the event loop (and therefore the SSE stream).
        """
        return await asyncio.to_thread(self.compute_flow, frame_t, frame_t1)

    def compute_batch_flows(
        self,
        frames: list[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Compute (fwd_flow, bwd_flow) for every consecutive pair in `frames`.

        Returns a list of length len(frames)-1.
        The i-th element is the flow between frames[i] and frames[i+1].

        Batching here means one RAFT inference per pair (RAFT is recurrent so
        true batch parallelism requires matching resolutions, which isn't
        guaranteed).  ProPainter and E2FGVI accept the resulting flow list
        directly.
        """
        self._ensure_loaded()
        results = []
        for i in range(len(frames) - 1):
            fwd, bwd = self.compute_flow(frames[i], frames[i + 1])
            results.append((fwd, bwd))
        return results

    async def compute_batch_flows_async(
        self,
        frames: list[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Async version of compute_batch_flows."""
        return await asyncio.to_thread(self.compute_batch_flows, frames)

    # ── Model loading ──────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load the RAFT model from checkpoint if not already loaded."""
        if self._model is not None:
            return

        _maybe_download_weights(self._checkpoint_path)

        try:
            import sys
            import os
            
            # Point Python directly to the core folder inside your RAFT directory
            raft_core_path = os.path.abspath(os.path.join(os.getcwd(), "RAFT", "core"))
            if raft_core_path not in sys.path:
                sys.path.append(raft_core_path)
            
            from raft import RAFT
        
        except ImportError as exc:
            raise ImportError(
                "RAFT is not installed.  Run:\n"
                "  git clone https://github.com/princeton-vl/RAFT.git\n"
                "  pip install -e RAFT/\n"
                f"Original error: {exc}"
            ) from exc

        import argparse
        args = argparse.Namespace(
            small           = False,   # full RAFT, not RAFT-Small
            mixed_precision = False,
            alternate_corr  = False,
        )

        model = RAFT(args)

        # Load weights — handle both raw state dict and DataParallel-wrapped files
        state = torch.load(str(self._checkpoint_path), map_location=self._device)
        if "model" in state:
            state = state["model"]
        # Strip "module." prefix added by DataParallel training
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(self._device)
        model.eval()

        self._model = model
        logger.info(
            "RAFT loaded: checkpoint=%s  device=%s",
            self._checkpoint_path.name, self._device,
        )


# ── Preprocessing helpers ──────────────────────────────────────────────────────

def _frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert a uint8 numpy frame (H, W, 3) to a float32 tensor [1, 3, H, W]
    in the range [0, 255] expected by RAFT (not normalised to [0, 1]).
    Accepts both BGR (OpenCV default) and RGB inputs — RAFT is flow-only so
    colour space doesn't affect the displacement vectors.
    """
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    t = torch.from_numpy(frame).permute(2, 0, 1).float()   # (3, H, W)
    return t.unsqueeze(0).to(device)                         # (1, 3, H, W)


def _flow_tensor_to_numpy(flow: torch.Tensor) -> np.ndarray:
    """
    Convert a RAFT output flow tensor [1, 2, H, W] to a numpy array (H, W, 2).
    """
    return flow.squeeze(0).permute(1, 2, 0).cpu().numpy()   # (H, W, 2)


class _InputPadder:
    """
    Pads images to dimensions that are multiples of 8.
    RAFT's correlation layers require this for the feature pyramid to work
    without remainder issues.

    Mirrors the InputPadder in the official RAFT repo's core/utils/utils.py
    so we don't depend on that utils module directly (import paths vary by
    installation method).
    """

    def __init__(self, shape: tuple, mode: str = "sintel") -> None:
        self._ht, self._wd = shape[-2], shape[-1]
        pad_ht = (((self._ht // 8) + 1) * 8 - self._ht) % 8
        pad_wd = (((self._wd // 8) + 1) * 8 - self._wd) % 8
        if mode == "sintel":
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs: torch.Tensor) -> list[torch.Tensor]:
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x: torch.Tensor) -> torch.Tensor:
        ht, wd = x.shape[-2], x.shape[-1]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


# ── Weight download helper ─────────────────────────────────────────────────────

def _maybe_download_weights(checkpoint_path: Path) -> None:
    """
    Download the RAFT-things checkpoint if it doesn't exist locally.
    Shows download progress in the logger.
    """
    if checkpoint_path.exists():
        return

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "RAFT checkpoint not found at %s — downloading from %s …",
        checkpoint_path, _RAFT_CHECKPOINT_URL,
    )

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0 and block_num % 50 == 0:
            pct = min(100, downloaded * 100 // total_size)
            logger.info("  Downloading RAFT checkpoint … %d%%", pct)

    try:
        urllib.request.urlretrieve(
            _RAFT_CHECKPOINT_URL, str(checkpoint_path), reporthook=_reporthook,
        )
        logger.info("RAFT checkpoint downloaded to %s", checkpoint_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download RAFT checkpoint: {exc}\n"
            f"Download manually from {_RAFT_CHECKPOINT_URL}\n"
            f"and place it at {checkpoint_path}"
        ) from exc
