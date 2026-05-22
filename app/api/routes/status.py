# ==============================================================================
# app/api/routes/status.py
#
# GET /status/{job_id}
#
# Opens an SSE (Server-Sent Events) stream for the given job.
# The frontend EventSource in script.js connects here immediately after
# POST /process returns, and listens until it receives a "complete" or
# "error" event.
#
# Event flow:
#   orchestrator.py  →  job_store.emit()  →  job_store._queues[job_id]
#                    →  job_store.listen() async generator
#                    →  this route's event_generator()
#                    →  sse-starlette EventSourceResponse
#                    →  browser EventSource object in script.js
#
# Late-join handling:
#   If a client connects after the pipeline is already COMPLETE or ERROR,
#   we immediately emit the terminal event and close the stream.  This
#   handles page-refresh scenarios without requiring the client to poll.
#
# Keep-alive:
#   If no event arrives within 25 s the generator yields a keep-alive comment.
#   Nginx and AWS ALB close idle SSE connections after 60 s by default;
#   25 s is safely below that threshold.
# ==============================================================================

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from app.api.dependencies import require_job
from app.core.job_store import job_store
from app.schemas.job import JobResult, JobStage, JobStatus

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/status/{job_id}",
    summary="Subscribe to pipeline progress via Server-Sent Events",
    description=(
        "Returns a text/event-stream response.  The browser EventSource "
        "receives 'progress' events during processing and a final 'complete' "
        "or 'error' event when the pipeline finishes."
    ),
)
async def status_stream(
    request:  Request,
    job: JobStatus = Depends(require_job),
) -> EventSourceResponse:

    job_id = job.job_id

    async def event_generator():
        # ── Late-join: job already finished ───────────────────────────────────
        # Re-fetch the job; the background task may have finished between the
        # Depends call above and now.
        current = await job_store.get(job_id)

        if current and current.stage == JobStage.COMPLETE:
            logger.debug("Late-join COMPLETE for job %s", job_id)
            yield {
                "event": "complete",
                "data": json.dumps({
                    "output_path": current.output_path,
                    "duration_s":  current.duration_s,
                    "tool":        current.tool,
                }),
            }
            return

        if current and current.stage == JobStage.ERROR:
            logger.debug("Late-join ERROR for job %s", job_id)
            yield {
                "event": "error",
                "data": json.dumps({"message": current.error or "Unknown pipeline error"}),
            }
            return

        # ── Normal path: stream events as they arrive ──────────────────────────
        async for event in job_store.listen(job_id):
            # Client disconnected — stop generating to free resources
            if await request.is_disconnected():
                logger.info("Client disconnected from SSE stream for job %s", job_id)
                break

            # Keep-alive sentinel from job_store.listen() — not a real event
            if event.get("__keepalive__"):
                # sse-starlette treats a dict with only "comment" as a comment line
                yield {"comment": "keep-alive"}
                continue

            # Extract the event name so sse-starlette sets `event: <name>`
            # separately from the JSON data payload.
            event_name = event.pop("event", "progress")

            yield {
                "event": event_name,
                "data":  json.dumps(event),
            }

            # After a terminal event the generator stops; the EventSource
            # on the client side receives readyState CLOSED automatically.
            if event_name in ("complete", "error"):
                logger.info("SSE stream closed for job %s (event=%s)", job_id, event_name)
                return

    return EventSourceResponse(event_generator())
