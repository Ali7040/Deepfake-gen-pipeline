"""Pydantic request/response schemas (the API contract)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ── Auth ─────────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    name: str = ""


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: EmailStr
    name: str
    created_at: datetime


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserOut


class RefreshRequest(BaseModel):
    refresh_token: str


class AccessToken(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Detection ────────────────────────────────────────────────────────────────
class FaceResult(BaseModel):
    label: str                       # "real" | "fake"
    real_confidence: int             # 0..100
    fake_confidence: int             # 0..100
    threat: str
    image_url: str | None = None     # cropped face evidence


class ImageDetectionResult(BaseModel):
    id: int | None = None
    media_type: str = "image"
    prediction: str                  # overall "real" | "fake"
    confidence: int                  # overall fake confidence 0..100
    face_count: int
    faces: list[FaceResult]
    processed_image_url: str | None = None
    original_image_url: str | None = None
    timing_ms: dict[str, int] | None = None
    created_at: datetime | None = None


class FrameResult(BaseModel):
    frame_number: int
    label: str
    real_confidence: int
    fake_confidence: int
    evidence_url: str | None = None


class VideoDetectionResult(BaseModel):
    id: int | None = None
    media_type: str = "video"
    prediction: str                  # "real" | "fake"
    confidence: int                  # fake-ratio percentage 0..100
    total_analyzed_frames: int
    fake_frames_detected: int
    frames: list[FrameResult]
    original_video_url: str | None = None
    timing_ms: dict[str, int] | None = None
    created_at: datetime | None = None


class ReportRequest(BaseModel):
    fake_confidence: int = Field(ge=0, le=100)
    record_id: int | None = None     # optionally attach to a history record


class ReportResult(BaseModel):
    success: bool
    report: str


class HistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    media_type: str
    verdict: str
    confidence: float
    original_file: str
    processed_file: str
    report: str
    created_at: datetime


# ── Generation proxy ─────────────────────────────────────────────────────────
class GenSwapResult(BaseModel):
    """Loose passthrough of the swap engine's response."""
    model_config = ConfigDict(extra="allow")
    success: bool = True
    job_id: str | None = None
