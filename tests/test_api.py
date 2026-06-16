# ==============================================================================
# tests/test_api.py
# Integration tests for FastAPI routes — no GPU, no real models.
# Uses FastAPI TestClient and mocks the pipeline orchestrator.
# ==============================================================================

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_test_video


@pytest.fixture(scope="module")
def client():
    """
    TestClient wired to a temporary storage directory.
    The pipeline orchestrator's run_pipeline is mocked so no GPU is needed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        for sub in ("uploads", "frames", "processed", "outputs"):
            os.makedirs(f"{tmpdir}/{sub}", exist_ok=True)

        env = {
            "UPLOAD_DIR":    f"{tmpdir}/uploads",
            "FRAMES_DIR":    f"{tmpdir}/frames",
            "PROCESSED_DIR": f"{tmpdir}/processed",
            "OUTPUT_DIR":    f"{tmpdir}/outputs",
        }
        with patch.dict(os.environ, env):
            import importlib
            import app.core.config as cfg_mod
            importlib.reload(cfg_mod)

            from main import app as fastapi_app
            with TestClient(fastapi_app, raise_server_exceptions=True) as c:
                yield c


# ── GET /health ────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_gpu_field_present(self, client):
        r = client.get("/health")
        assert "gpu_available" in r.json()


# ── POST /upload ───────────────────────────────────────────────────────────────

class TestUpload:
    def _make_mp4_bytes(self) -> bytes:
        """Create a minimal MP4 in memory using a temp file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            make_test_video(Path(tf.name), n_frames=8, fps=8, width=64, height=64)
            return Path(tf.name).read_bytes()

    def test_upload_returns_job_id(self, client):
        mp4 = self._make_mp4_bytes()
        r   = client.post("/upload", files={"file": ("test.mp4", mp4, "video/mp4")})
        assert r.status_code == 200
        body = r.json()
        assert "job_id" in body
        assert len(body["job_id"]) == 8

    def test_upload_returns_metadata(self, client):
        mp4 = self._make_mp4_bytes()
        r   = client.post("/upload", files={"file": ("test.mp4", mp4, "video/mp4")})
        body = r.json()
        assert body.get("fps") is not None
        assert body.get("width") is not None
        assert body.get("height") is not None

    def test_unsupported_format_returns_422(self, client):
        r = client.post("/upload", files={"file": ("bad.txt", b"hello", "text/plain")})
        assert r.status_code == 422

    def test_upload_registers_job(self, client):
        mp4 = self._make_mp4_bytes()
        r   = client.post("/upload", files={"file": ("test.mp4", mp4, "video/mp4")})
        job_id = r.json()["job_id"]
        # Job should be findable via /status (even though pipeline hasn't started)
        with client.stream("GET", f"/status/{job_id}") as sse:
            assert sse.status_code == 200  # connection opens successfully

    def test_unknown_job_404(self, client):
        with client.stream("GET", "/status/deadbeef") as sse:
            assert sse.status_code == 404


# ── POST /process ──────────────────────────────────────────────────────────────

class TestProcess:
    def _upload(self, client) -> str:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            make_test_video(Path(tf.name), n_frames=8, fps=8, width=64, height=64)
            mp4 = Path(tf.name).read_bytes()
        r = client.post("/upload", files={"file": ("test.mp4", mp4, "video/mp4")})
        return r.json()["job_id"]

    def test_process_returns_job_id(self, client):
        job_id = self._upload(client)
        with patch("app.pipeline.orchestrator.run_pipeline", new_callable=AsyncMock):
            r = client.post("/process", json={
                "job_id": job_id,
                "tool":   "object_removal",
                "params": {
                    "prompt":             "the cat",
                    "detection_backend":  "grounding_dino",
                    "inpaint_model":      "propainter",
                    "temporal_smoothing": True,
                    "mask_dilation_px":   8,
                },
            })
        assert r.status_code == 200
        assert r.json()["job_id"] == job_id

    def test_process_invalid_tool_returns_422(self, client):
        job_id = self._upload(client)
        r = client.post("/process", json={
            "job_id": job_id,
            "tool":   "nonexistent_tool",
            "params": {"prompt": "x"},
        })
        assert r.status_code == 422

    def test_process_unknown_job_returns_404(self, client):
        r = client.post("/process", json={
            "job_id": "00000000",
            "tool":   "object_removal",
            "params": {"prompt": "x", "detection_backend": "grounding_dino",
                       "inpaint_model": "propainter",
                       "temporal_smoothing": True, "mask_dilation_px": 8},
        })
        assert r.status_code == 404

    def test_style_transfer_params_validated(self, client):
        job_id = self._upload(client)
        with patch("app.pipeline.orchestrator.run_pipeline", new_callable=AsyncMock):
            r = client.post("/process", json={
                "job_id": job_id,
                "tool":   "style_transfer",
                "params": {
                    "prompt":              "snowy winter",
                    "style_backbone":      "controlnet",
                    "style_strength":      0.75,
                    "flow_guidance":       True,
                    "flicker_suppression": True,
                    "inference_steps":     25,
                },
            })
        assert r.status_code == 200


# ── Pydantic schema tests ──────────────────────────────────────────────────────

class TestSchemas:
    def test_object_removal_params_missing_prompt(self):
        from pydantic import ValidationError
        from app.schemas.requests import ObjectRemovalParams
        with pytest.raises(ValidationError):
            ObjectRemovalParams()   # prompt is required

    def test_style_strength_out_of_range(self):
        from pydantic import ValidationError
        from app.schemas.requests import StyleTransferParams
        with pytest.raises(ValidationError):
            StyleTransferParams(prompt="x", style_strength=1.5)   # max 1.0

    def test_mask_dilation_out_of_range(self):
        from pydantic import ValidationError
        from app.schemas.requests import ObjectRemovalParams
        with pytest.raises(ValidationError):
            ObjectRemovalParams(prompt="x", mask_dilation_px=100)  # max 32
