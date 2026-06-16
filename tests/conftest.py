# ==============================================================================
# tests/conftest.py — Shared pytest fixtures
# ==============================================================================

from __future__ import annotations

import asyncio
import io
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


# ── Synthetic video generator ──────────────────────────────────────────────────

def make_test_video(
    path:    Path,
    n_frames: int  = 24,
    fps:      int  = 24,
    width:    int  = 320,
    height:   int  = 240,
) -> Path:
    """
    Write a synthetic colour-gradient MP4 to `path`.
    Each frame has a different hue so motion is clearly visible.
    No external assets needed — pure OpenCV.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for i in range(n_frames):
        # HSV gradient: hue sweeps 0→180 across frames
        hue   = int(180 * i / n_frames)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = hue           # H
        frame[:, :, 1] = 200           # S
        frame[:, :, 2] = 200           # V
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        out.write(frame)

    out.release()
    return path


def make_frames(n: int = 8, h: int = 64, w: int = 64) -> list[np.ndarray]:
    """Return `n` random uint8 BGR frames at (h, w)."""
    return [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def make_masks(n: int = 8, h: int = 64, w: int = 64) -> list[np.ndarray]:
    """Return `n` binary uint8 masks (255/0) with a centred rectangle masked."""
    masks = []
    for _ in range(n):
        m = np.zeros((h, w), dtype=np.uint8)
        m[h//4: 3*h//4, w//4: 3*w//4] = 255
        masks.append(m)
    return masks


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("flowedit_tests")


@pytest.fixture(scope="session")
def test_video_path(tmp_dir):
    """A real 24-frame synthetic MP4 file on disk."""
    p = tmp_dir / "test_video.mp4"
    return make_test_video(p)


@pytest.fixture
def sample_frames():
    return make_frames()


@pytest.fixture
def sample_masks():
    return make_masks()


@pytest.fixture
def mock_flow_engine():
    """Mock RAFTFlowEngine — returns zero-displacement flow (identity warp)."""
    engine = MagicMock()
    h, w   = 64, 64
    zero   = np.zeros((h, w, 2), dtype=np.float32)
    engine.compute_flow.return_value = (zero, zero)
    engine.compute_flow_async = AsyncMock(return_value=(zero, zero))
    engine.compute_batch_flows.return_value = [(zero, zero)] * 7
    engine.compute_batch_flows_async = AsyncMock(return_value=[(zero, zero)] * 7)
    return engine


@pytest.fixture
def app_client():
    """FastAPI TestClient with all storage dirs in a tmp location."""
    import os, sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    with tempfile.TemporaryDirectory() as tmpdir:
        env_patch = {
            "UPLOAD_DIR":    f"{tmpdir}/uploads",
            "FRAMES_DIR":    f"{tmpdir}/frames",
            "PROCESSED_DIR": f"{tmpdir}/processed",
            "OUTPUT_DIR":    f"{tmpdir}/outputs",
        }
        with patch.dict(os.environ, env_patch):
            # Re-import settings so the patched env vars take effect
            import importlib
            import app.core.config as _cfg
            importlib.reload(_cfg)

            from main import app
            for d in env_patch.values():
                os.makedirs(d, exist_ok=True)

            client = TestClient(app)
            yield client

    # Restore settings singleton
    import app.core.config as _cfg
    importlib.reload(_cfg)


import sys

@pytest.fixture(scope="session")
def event_loop():
    """Force pytest to use a clean, stable event loop provider on Windows."""
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()