# ==============================================================================
# FastAPI application entry point.
# Responsibilities:
#   • Creates all storage/ subdirectories on startup (lifespan)
#   • Adds CORS middleware (origins from settings.cors_origins)
#   • Mounts storage/outputs/ as a static file directory → GET /outputs/ {file}
#   • Registers the three API routers (upload, process, status)
#   • Exposes GET /health for the frontend server-status indicatorr
# ==============================================================================

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import process, status, upload
from app.core.config import settings

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = getattr(logging, settings.log_level.upper(), logging.INFO),
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on server startup (before yield) and shutdown (after yield).

    Startup:
      • Creates the four storage directories if they don't exist.
        This means a fresh clone needs no manual mkdir commands — just
        `uvicorn main:app` and everything is ready.

    Shutdown:
      • (nothing yet — add model teardown / file cleanup here in Step 10)
    """
    dirs = [
        settings.upload_dir,
        settings.frames_dir,
        settings.processed_dir,
        settings.output_dir,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.debug("Storage directory ready: %s", d)

    gpu = torch.cuda.is_available()
    logger.info(
        "FlowEdit starting — GPU: %s  Replicate fallback: %s",
        torch.cuda.get_device_name(0) if gpu else "not available",
        settings.use_replicate_fallback,
    )

    yield  # server is running

    logger.info("FlowEdit shutting down")


# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "FlowEdit API",
    description = "Generative AI video editing platform — backend API",
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",   # Swagger UI
    redoc_url   = "/redoc",  # ReDoc UI
)


# ── CORS ───────────────────────────────────────────────────────────────────────
# The frontend is served from a different origin (e.g. python -m http.server
# on port 5500) so we need permissive CORS in development.
# In production, restrict allow_origins to your actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.cors_origins,
    allow_credentials = True,
    allow_methods     = ["GET", "POST", "OPTIONS"],
    allow_headers     = ["*"],
    expose_headers    = ["Content-Type"],
)


# ── Static files ───────────────────────────────────────────────────────────────
# Processed videos are served directly from disk.
# The frontend download button hits GET /outputs/{job_id}.mp4.
# html=False means 404 (not a directory listing) for unknown paths.
app.mount(
    "/outputs",
    StaticFiles(directory=settings.output_dir, html=False),
    name="outputs",
)


# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(upload.router,  tags=["Video Upload"])
app.include_router(process.router, tags=["Pipeline"])
app.include_router(status.router,  tags=["Progress"])


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"], summary="Server and GPU status")
async def health() -> dict:
    """
    Polled by the frontend every 30 s to update the status indicator.
    Returns GPU availability so the UI can suggest enabling the Replicate
    toggle when no local GPU is detected.
    """
    gpu_available = torch.cuda.is_available()
    return {
        "status":        "ok",
        "gpu_available": gpu_available,
        "gpu_name":      torch.cuda.get_device_name(0) if gpu_available else None,
    }
