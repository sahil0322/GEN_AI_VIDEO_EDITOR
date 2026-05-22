# ==============================================================================
# app/api/dependencies.py
#
# Shared FastAPI Depends — injected into any route that needs a validated
# job_id.  Raises 404 before the route body runs if the job doesn't exist,
# keeping validation logic out of each individual route handler.
#
# Usage in a route:
#   @router.get("/status/{job_id}")
#   async def status(job_id: str = Depends(require_job)):
#       ...
# ==============================================================================

from __future__ import annotations

from fastapi import Depends, HTTPException, Path

from app.core.job_store import job_store
from app.schemas.job import JobStatus


async def require_job(
    job_id: str = Path(..., description="Job ID returned by POST /upload"),
) -> JobStatus:
    """
    Resolve a job_id path parameter to its JobStatus object.

    Raises:
        HTTPException 404: if job_id is not in the store.

    Returns:
        The live JobStatus for the given job.
    """
    job = await job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. "
                   "Upload a video first via POST /upload.",
        )
    return job
