# ==============================================================================
# app/api/routes/upload.py
#
# POST /upload
#
# Accepts a multipart video file, validates it, saves it to disk, extracts
# video metadata with OpenCV, registers the job in job_store, and returns
# a JSON payload the frontend uses to populate the metadata row and enable
# the process button.
#
# Pipeline hook:
#   The file is saved as  storage/uploads/{job_id}/original{.mp4|.mov|.webm}
#   The extractor (Step 4) reads from this exact path when the pipeline runs.
# ==============================================================================

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import aiofiles
import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings
from app.core.job_store import job_store
from app.schemas.job import JobStatus
from app.schemas.requests import UploadResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Allowed file extensions (content-type headers from browsers are unreliable
# for video so we validate by extension too)
_ALLOWED_SUFFIXES = {".mp4", ".mov", ".webm"}


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a video file",
    description=(
        "Accepts MP4, MOV, or WebM. Max size is controlled by MAX_UPLOAD_MB "
        "in .env (default 500 MB). Returns a job_id and video metadata."
    ),
)
async def upload_video(file: UploadFile = File(...)) -> UploadResponse:
    # ── 1. Validate extension ──────────────────────────────────────────────────
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file format '{suffix}'. "
                f"Accepted formats: {', '.join(sorted(_ALLOWED_SUFFIXES))}"
            ),
        )

    # ── 2. Read and size-check ─────────────────────────────────────────────────
    # We read the whole file into memory before writing to disk so we can check
    # the size without a partial write.  At ≤500 MB this is acceptable.
    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({len(content) / 1_048_576:.1f} MB). "
                f"Maximum allowed: {settings.max_upload_mb} MB."
            ),
        )

    # ── 3. Create job directory and save file ──────────────────────────────────
    job_id  = uuid.uuid4().hex[:8]           # short enough for URLs, unique enough for dev
    job_dir = Path(settings.upload_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    save_path = job_dir / f"original{suffix}"
    async with aiofiles.open(save_path, "wb") as fh:
        await fh.write(content)

    logger.info("Saved upload: job=%s  file=%s  size=%.1f MB", job_id, file.filename, len(content) / 1_048_576)

    # ── 4. Extract video metadata with OpenCV ──────────────────────────────────
    # Metadata is best-effort: if OpenCV can't open the file (e.g. unusual
    # codec) we still register the job and return None for the fields.
    # The frontend shows "—" for missing metadata, which is acceptable.
    duration_s: float | None = None
    fps:        float | None = None
    width:      int   | None = None
    height:     int   | None = None

    try:
        cap = cv2.VideoCapture(str(save_path))
        if cap.isOpened():
            raw_fps    = cap.get(cv2.CAP_PROP_FPS)
            raw_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            raw_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            raw_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if raw_fps and raw_fps > 0:
                fps        = round(raw_fps, 2)
                duration_s = round(raw_frames / raw_fps, 2) if raw_frames > 0 else None

            width  = raw_w if raw_w > 0 else None
            height = raw_h if raw_h > 0 else None

        cap.release()
    except Exception as exc:
        logger.warning("OpenCV metadata extraction failed for job %s: %s", job_id, exc)

    # ── 5. Register job in the store ───────────────────────────────────────────
    status = JobStatus(
        job_id     = job_id,
        filename   = file.filename or f"upload{suffix}",
        duration_s = duration_s,
        fps        = fps,
        width      = width,
        height     = height,
    )
    await job_store.create(job_id, status)

    return UploadResponse(
        job_id     = job_id,
        filename   = file.filename or f"upload{suffix}",
        duration_s = duration_s,
        fps        = fps,
        width      = width,
        height     = height,
    )
