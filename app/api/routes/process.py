# ==============================================================================
# app/api/routes/process.py
#
# POST /process
#
# Validates the ProcessRequest (including tool-specific params via the
# model_validator in requests.py), enqueues the pipeline as a FastAPI
# BackgroundTask, and returns immediately with {"job_id": "..."}.
#
# The frontend then opens GET /status/{job_id} to receive SSE progress events.
#
# Design note — BackgroundTasks vs asyncio.create_task:
#   FastAPI BackgroundTasks run after the response is sent but are still
#   managed by the same event loop.  For a long-running inference pipeline
#   this is fine because all the heavy work (torch inference, FFmpeg) is
#   either run in a thread executor (see orchestrator.py) or is truly async.
#   For multi-worker production deployments, replace with Celery or
#   ARQ (async Redis queue) without changing any route code.
# ==============================================================================

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.api.dependencies import require_job
from app.core.job_store import job_store
from app.schemas.job import JobStatus
from app.schemas.requests import ProcessRequest, ProcessResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/process",
    response_model=ProcessResponse,
    summary="Start the AI pipeline for an uploaded video",
    description=(
        "Validates tool params, enqueues the pipeline as a background task, "
        "and returns immediately.  Subscribe to GET /status/{job_id} for "
        "real-time progress via Server-Sent Events."
    ),
)
async def process_video(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
) -> ProcessResponse:
    # Call the dependency manually using the ID from the JSON body
    job = await require_job(job_id=request.job_id)

    # Guard against re-processing an already-running or completed job
    from app.schemas.job import JobStage
    if job.stage not in (JobStage.PENDING, JobStage.ERROR):
        raise HTTPException(
            409,
            detail=(
                f"Job '{request.job_id}' is already in stage '{job.stage.value}'. "
                "Upload a new video to start a fresh job."
            ),
        )

    # Persist the tool choice onto the job status so /status can reflect it
    await job_store.update(
        request.job_id,
        tool          = request.tool,
        use_replicate = request.use_replicate,
    )

    # ── Enqueue pipeline ───────────────────────────────────────────────────────
    # Import deferred so heavy model imports don't block the server startup.
    # The orchestrator is the only caller of the AI model stack.
    from app.pipeline.orchestrator import run_pipeline

    background_tasks.add_task(
        run_pipeline,
        job_id        = request.job_id,
        tool          = request.tool,
        params        = request.params,          # already validated + normalised dict
        use_replicate = request.use_replicate,
    )

    logger.info(
        "Enqueued pipeline: job=%s  tool=%s  replicate=%s",
        request.job_id, request.tool, request.use_replicate,
    )

    return ProcessResponse(job_id=request.job_id)


# ── Override require_job to use body job_id (not a path param) ────────────────
# FastAPI's Depends resolves Path params by name.  Since /process has no
# {job_id} path segment we override require_job inline here so it reads from
# the request body instead.
#
# We achieve this by declaring the dependency differently from the normal
# path-param version: extract job_id from the body model before calling
# job_store.get(), keeping require_job in dependencies.py usable for routes
# that DO have {job_id} as a path segment (like /status/{job_id}).
