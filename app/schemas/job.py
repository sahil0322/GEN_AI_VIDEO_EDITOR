# ==============================================================================
# app/schemas/job.py
#
# Pydantic models representing the state of a pipeline job.
# These are stored in job_store._jobs and serialised into SSE event payloads.
# ==============================================================================

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStage(str, Enum):
    """
    Ordered pipeline stages.

    The orchestrator advances through these in sequence.
    The SSE /status route uses the string values as the "stage" field in
    progress events, which the frontend maps to its four pipeline-dot elements:
        extract → pStage1
        batch   → pStage2
        infer   → pStage3
        recompose → pStage4
    """
    PENDING   = "pending"    # job registered, pipeline not started yet
    EXTRACT   = "extract"    # Stage 1: FFmpeg frame extraction
    BATCH     = "batch"      # Stage 2: grouping frames into batches
    INFER     = "infer"      # Stage 3: AI model inference
    RECOMPOSE = "recompose"  # Stage 4: FFmpeg stitch + audio mux
    COMPLETE  = "complete"   # pipeline finished successfully
    ERROR     = "error"      # pipeline failed


class JobStatus(BaseModel):
    """
    Mutable state object for a single processing job.
    Stored in the JobStore and patched via job_store.update().
    """
    job_id:   str
    filename: str

    # Pipeline state
    stage:    JobStage = JobStage.PENDING
    progress: float    = Field(0.0, ge=0, le=100)  # 0–100
    message:  str      = "Queued"

    # Set by /upload (read from OpenCV metadata)
    duration_s: Optional[float] = None
    fps:        Optional[float] = None
    width:      Optional[int]   = None
    height:     Optional[int]   = None

    # Set by /process
    tool:          Optional[str] = None  # "object_removal" | "style_transfer"
    use_replicate: bool = False

    # Set on completion
    output_path: Optional[str] = None

    # Set on error
    error: Optional[str] = None


class JobResult(BaseModel):
    """
    Payload of the SSE "complete" event emitted by the orchestrator.
    Also returned if a client polls /status after the job is already done.
    """
    job_id:      str
    output_path: str           # relative path, e.g. "storage/outputs/a3f7c2d1.mp4"
    duration_s:  float
    tool:        str           # "object_removal" | "style_transfer"
