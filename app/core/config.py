# ==============================================================================
# app/core/config.py
#
# All configuration is read from .env (or real environment variables) via
# Pydantic BaseSettings. Importing `settings` anywhere gives you a single
# validated, type-safe config object with no global mutable state.
#
# Usage:
#   from app.core.config import settings
#   batch_size = settings.infer_batch_size
# ==============================================================================

from __future__ import annotations

from typing import List
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    # ── Server ─────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # Accept origins as a comma-separated string from the env var,
    # e.g.  CORS_ORIGINS="http://localhost:5500,http://localhost:3000"
    cors_origins: List[str] = [
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors(cls, v):
        """Allow CORS_ORIGINS env var to be a comma-separated string."""
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    # ── Storage paths ──────────────────────────────────────────────────────────
    upload_dir:    str = "storage/uploads"    # raw MP4s
    frames_dir:    str = "storage/frames"     # extracted PNGs  (job_id/frame_NNNN.png)
    processed_dir: str = "storage/processed"  # AI-output PNGs  (job_id/frame_NNNN.png)
    output_dir:    str = "storage/outputs"    # final MP4s      (job_id.mp4)
    max_upload_mb: int = 500

    # ── Model weight file paths ────────────────────────────────────────────────
    # SAM (Segment Anything Model)
    sam_checkpoint:  str = "weights/sam_vit_h_4b8939.pth"
    sam_model_type:  str = "vit_h"

    # GroundingDINO
    grounding_dino_config:  str = (
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    grounding_dino_weights: str = "weights/groundingdino_swint_ogc.pth"

    # RAFT optical flow (shared by both tools — see pipeline/temporal/optical_flow.py)
    raft_checkpoint: str = "weights/raft-things.pth"

    # ProPainter
    propainter_checkpoint: str = "weights/ProPainter.pth"

    # ── Batch sizes (tune to available VRAM) ───────────────────────────────────
    # infer_batch_size: number of frames per inference batch sent to each model.
    # raft_batch_size : number of consecutive frame *pairs* fed to RAFT at once.
    # Lower these values if you hit CUDA OOM; the pipeline is stateless between
    # batches so results are identical regardless of batch size.
    infer_batch_size: int = 8
    raft_batch_size:  int = 4

    # ── Replicate cloud GPU fallback ───────────────────────────────────────────
    replicate_api_token:    str  = ""
    use_replicate_fallback: bool = False

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Single shared instance — import this everywhere.
# Constructed once at module load time; all values are read-only after that.
settings = Settings()
