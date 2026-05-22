# ==============================================================================
# app/core/job_store.py
#
# Thread-safe in-memory store that holds:
#   • JobStatus objects  (current state of every job)
#   • asyncio.Queue     (per-job SSE event bus)
#
# Architecture note:
#   The orchestrator pushes progress events to the queue via emit().
#   The /status/{job_id} SSE route consumes those events via listen().
#   This decouples the pipeline from the transport layer — swapping
#   the queue for Redis Streams later requires changing only this file.
#
# Swap path to Redis:
#   Replace self._queues with aioredis pub/sub channels.
#   emit()  → channel.publish(job_id, json.dumps(event))
#   listen() → channel.subscribe(job_id) async generator
#   No route or orchestrator code needs to change.
# ==============================================================================

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, Dict, Optional

from app.schemas.job import JobStage, JobStatus

logger = logging.getLogger(__name__)

# Sentinel pushed to the queue by the orchestrator when the pipeline is done
# (whether complete or errored) so the SSE generator knows to close the stream.
_STREAM_DONE = object()


class JobStore:
    """
    Singleton in-memory store.

    All mutating methods are protected by an asyncio.Lock so they are safe to
    call from multiple coroutines (e.g. the background pipeline task and the
    SSE status route running concurrently in the same event loop).
    """

    def __init__(self) -> None:
        self._jobs:   Dict[str, JobStatus]     = {}
        self._queues: Dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def create(self, job_id: str, status: JobStatus) -> None:
        """Register a new job with an empty event queue."""
        async with self._lock:
            self._jobs[job_id]   = status
            self._queues[job_id] = asyncio.Queue()
        logger.debug("Created job %s", job_id)

    def exists(self, job_id: str) -> bool:
        return job_id in self._jobs

    async def get(self, job_id: str) -> Optional[JobStatus]:
        return self._jobs.get(job_id)

    async def update(self, job_id: str, **kwargs) -> None:
        """Patch any fields on a JobStatus in-place."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                logger.warning("update() called on unknown job %s", job_id)
                return
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
                else:
                    logger.warning("JobStatus has no field '%s'", key)

    # ── Event bus ──────────────────────────────────────────────────────────────

    async def emit(self, job_id: str, event: dict) -> None:
        """
        Push an SSE event dict onto the job's queue.

        The orchestrator calls this after every batch. The event dict must
        contain an "event" key ("progress", "complete", or "error").

        Example:
            await job_store.emit(job_id, {
                "event":   "progress",
                "stage":   "infer",
                "pct":     42,
                "message": "Processing frame 50 / 120",
                "detail":  "Running ProPainter batch 6 / 15",
            })
        """
        queue = self._queues.get(job_id)
        if queue is None:
            logger.warning("emit() called for unknown job %s", job_id)
            return
        await queue.put(event)

    async def close_stream(self, job_id: str) -> None:
        """
        Signal to the SSE generator that no more events will be emitted.
        Called by the orchestrator after emitting the final "complete" or
        "error" event.
        """
        queue = self._queues.get(job_id)
        if queue is not None:
            await queue.put(_STREAM_DONE)

    async def listen(self, job_id: str) -> AsyncGenerator[dict, None]:
        """
        Async generator consumed by the SSE route.

        Yields event dicts until the pipeline signals completion or an error
        occurs. Yields a keep-alive sentinel every 25 seconds if no event
        arrives, preventing proxy/browser timeout.

        Usage (in the SSE route):
            async for event in job_store.listen(job_id):
                yield event
        """
        queue = self._queues.get(job_id)
        if queue is None:
            return

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=25.0)
            except asyncio.TimeoutError:
                # Yield a keep-alive ping — sse-starlette serialises
                # this as a comment line (": keep-alive\n\n") which
                # browsers ignore but proxies use to reset idle timers.
                yield {"__keepalive__": True}
                continue

            if event is _STREAM_DONE:
                return

            yield event

            # Stop iterating after terminal events so the generator closes
            # cleanly regardless of whether close_stream() was called.
            if event.get("event") in ("complete", "error"):
                return


# ── Singleton ──────────────────────────────────────────────────────────────────
# Import this instance everywhere rather than instantiating JobStore directly.
job_store = JobStore()
