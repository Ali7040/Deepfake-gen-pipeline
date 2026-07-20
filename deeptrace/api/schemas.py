"""Pydantic schemas for DeepTrace API requests/responses."""
from __future__ import annotations

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class FaceCrop(BaseModel):
    b64: str
    score: float
    bbox: List[float]


class DetectFacesResponse(BaseModel):
    success: bool
    count: int = 0
    faces: List[FaceCrop] = []
    image_path: Optional[str] = None
    error: Optional[str] = None


class SwapResponse(BaseModel):
    success: bool
    job_id: Optional[str] = None
    output_filename: Optional[str] = None
    output_url: Optional[str] = None
    output_type: Optional[str] = None
    processing_time: Optional[float] = None
    faces_swapped: Optional[int] = None
    frames_processed: Optional[int] = None
    frames_skipped: Optional[int] = None
    fps_achieved: Optional[float] = None
    output_resolution: Optional[str] = None
    preview_b64: Optional[str] = None
    error: Optional[str] = None


class WebcamFrameRequest(BaseModel):
    frame_b64: str = Field(..., description="Base64-encoded JPEG frame")
    enhance: bool = False


class WebcamFrameResponse(BaseModel):
    success: bool
    result_b64: Optional[str] = None
    fps: Optional[float] = None
    faces_found: Optional[int] = None
    error: Optional[str] = None


class WebcamSourceResponse(BaseModel):
    success: bool
    preview_b64: Optional[str] = None
    score: Optional[float] = None
    error: Optional[str] = None


class ProgressResponse(BaseModel):
    found: bool
    total: Optional[int] = None
    done: Optional[int] = None
    status: Optional[str] = None
    eta_seconds: Optional[int] = None
    fps_proc: Optional[float] = None
    skipped: Optional[int] = None


class ActiveJobsResponse(BaseModel):
    jobs: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    init_ok: bool
    warmup_done: bool
    version: str
