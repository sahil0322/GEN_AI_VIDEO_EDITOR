# ==============================================================================
# app/schemas/requests.py
#
# Pydantic models for incoming HTTP request bodies and outgoing responses.
#
# ProcessRequest uses a model_validator to validate the `params` dict against
# the correct sub-model based on the `tool` field.  This keeps the API surface
# clean (one /process endpoint, not one per tool) while ensuring each tool's
# parameters are fully validated before the pipeline is enqueued.
# ==============================================================================

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ── Tool parameter schemas ─────────────────────────────────────────────────────

class ObjectRemovalParams(BaseModel):
    """
    Parameters for Tool 1: Semantic Object Removal.

    detection_backend controls whether the frontend's text prompt is used by
    GroundingDINO (text → bbox → SAM) or if the user drew a bounding box
    directly (bbox → SAM).

    temporal_smoothing:
        When True, mask_propagator.py uses RAFT optical flow to warp the
        frame-0 segmentation mask forward in time instead of re-running SAM on
        every frame.  This prevents the "mask jitter" artifact on fast camera
        moves and cuts inference time by ~70% (SAM runs only once).
    """
    prompt:             str   = Field(...,  description="Text describing the object to remove, e.g. 'the person on the left'")
    detection_backend:  Literal["grounding_dino", "sam_bbox"] = "grounding_dino"
    inpaint_model:      Literal["propainter", "e2fgvi"]       = "propainter"
    temporal_smoothing: bool  = True
    mask_dilation_px:   int   = Field(8, ge=0, le=32, description="Expand the mask outward by N pixels to avoid edge artefacts")


class StyleTransferParams(BaseModel):
    """
    Parameters for Tool 2: Prompt-Based Style Transfer.

    flow_guidance:
        When True, flow_guidance.py warps the previously generated styled
        frame using RAFT's forward-flow field and passes it as ControlNet
        depth conditioning for the next frame.  Without this each frame is
        styled independently and there is no inter-frame colour continuity.

    flicker_suppression:
        When True, flicker_suppressor.py computes a per-pixel flow-consistency
        score (from RAFT's forward+backward flow cycle-consistency) and uses
        it as a blend weight: pixels with high consistency (static background)
        keep the new diffusion output; pixels with low consistency (motion
        boundaries) blend toward the warped-previous frame.  This removes the
        strobing effect common in per-frame diffusion pipelines.
    """
    prompt:              str   = Field(..., description="Style description, e.g. 'a snowy winter landscape, cinematic'")
    style_backbone:      Literal["svd", "controlnet"] = "svd"
    style_strength:      float = Field(0.75, ge=0.1, le=1.0)
    flow_guidance:       bool  = True
    flicker_suppression: bool  = True
    inference_steps:     int   = Field(25, ge=10, le=50)


# ── Route request/response models ─────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Returned by POST /upload on success."""
    job_id:     str
    filename:   str
    duration_s: Optional[float] = None
    fps:        Optional[float] = None
    width:      Optional[int]   = None
    height:     Optional[int]   = None


class ProcessRequest(BaseModel):
    """
    Body of POST /process.

    `params` is validated against ObjectRemovalParams or StyleTransferParams
    depending on the value of `tool`.  After validation the dict is normalised
    so the orchestrator always receives well-typed kwargs.
    """
    job_id:        str
    tool:          Literal["object_removal", "style_transfer"]
    params:        Dict[str, Any]
    use_replicate: bool = False

    @model_validator(mode="after")
    def _validate_and_normalise_params(self) -> "ProcessRequest":
        """
        Validate `params` against the correct sub-schema and normalise the
        dict so the orchestrator never receives unexpected keys.
        """
        if self.tool == "object_removal":
            validated = ObjectRemovalParams(**self.params)
        else:
            validated = StyleTransferParams(**self.params)

        # Replace the raw dict with the validated, serialised version.
        # model_dump() excludes unset fields by default in Pydantic v2 which
        # is exactly what the orchestrator dispatch needs.
        self.params = validated.model_dump()
        return self


class ProcessResponse(BaseModel):
    """Returned by POST /process on success."""
    job_id: str
