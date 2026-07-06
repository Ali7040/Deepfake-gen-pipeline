"""ORM models: users and detection history."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), default="")
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    records: Mapped[list["DetectionRecord"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class DetectionRecord(Base):
    """One detection run (image or video) and its verdict."""

    __tablename__ = "detection_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )
    media_type: Mapped[str] = mapped_column(String(16))          # "image" | "video"
    verdict: Mapped[str] = mapped_column(String(16))             # "real" | "fake"
    confidence: Mapped[float] = mapped_column(Float, default=0.0)  # 0..100
    original_file: Mapped[str] = mapped_column(String(512), default="")
    processed_file: Mapped[str] = mapped_column(String(512), default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)     # per-face/frame data
    report: Mapped[str] = mapped_column(Text, default="")        # optional Qwen report
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    user: Mapped["User"] = relationship(back_populates="records")
