"""Native deepfake detection endpoints (image, video, forensic report, history)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user, get_current_user_optional
from app.models import DetectionRecord, User
from app.schemas import (
    HistoryItem,
    ImageDetectionResult,
    ReportRequest,
    ReportResult,
    VideoDetectionResult,
)
from app.services.detect_image import detect_image
from app.services.detect_video import detect_video
from app.services.report import generate_report
from app.utils import IMAGE_EXTS, VIDEO_EXTS, save_upload

router = APIRouter(prefix="/detection", tags=["detection"])


@router.post("/image", response_model=ImageDetectionResult)
async def detect_image_endpoint(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
) -> ImageDetectionResult:
    path = await save_upload(file, "img", IMAGE_EXTS)
    # Inference is CPU/GPU bound -> run off the event loop.
    result = await asyncio.to_thread(detect_image, path)
    if "error" in result:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, result["error"])

    record = DetectionRecord(
        user_id=user.id if user else None,
        media_type="image",
        verdict=result["prediction"],
        confidence=result["confidence"],
        original_file=f"/media/uploads/{path.name}",
        processed_file=result.get("processed_image_url", ""),
        detail={"faces": result["faces"]},
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return ImageDetectionResult(
        id=record.id,
        prediction=result["prediction"],
        confidence=result["confidence"],
        face_count=result["face_count"],
        faces=result["faces"],
        processed_image_url=result.get("processed_image_url"),
        original_image_url=f"/media/uploads/{path.name}",
        timing_ms=result.get("timing_ms"),
        created_at=record.created_at,
    )


@router.post("/video", response_model=VideoDetectionResult)
async def detect_video_endpoint(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
) -> VideoDetectionResult:
    path = await save_upload(file, "vid", VIDEO_EXTS)
    result = await asyncio.to_thread(detect_video, path)
    if "error" in result:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, result["error"])

    record = DetectionRecord(
        user_id=user.id if user else None,
        media_type="video",
        verdict=result["prediction"],
        confidence=result["confidence"],
        original_file=f"/media/uploads/{path.name}",
        detail={
            "frames": result["frames"],
            "total_analyzed_frames": result["total_analyzed_frames"],
            "fake_frames_detected": result["fake_frames_detected"],
        },
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return VideoDetectionResult(
        id=record.id,
        prediction=result["prediction"],
        confidence=result["confidence"],
        total_analyzed_frames=result["total_analyzed_frames"],
        fake_frames_detected=result["fake_frames_detected"],
        frames=result["frames"],
        original_video_url=f"/media/uploads/{path.name}",
        timing_ms=result.get("timing_ms"),
        created_at=record.created_at,
    )


@router.post("/report", response_model=ReportResult)
async def forensic_report(
    payload: ReportRequest, db: Session = Depends(get_db)
) -> ReportResult:
    text = await asyncio.to_thread(generate_report, payload.fake_confidence)
    if payload.record_id is not None:
        record = db.get(DetectionRecord, payload.record_id)
        if record is not None:
            record.report = text
            db.commit()
    return ReportResult(success=True, report=text)


@router.get("/history", response_model=list[HistoryItem])
def history(
    limit: int = 50,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[HistoryItem]:
    rows = db.scalars(
        select(DetectionRecord)
        .where(DetectionRecord.user_id == user.id)
        .order_by(DetectionRecord.created_at.desc())
        .limit(min(limit, 200))
    ).all()
    return [HistoryItem.model_validate(r) for r in rows]


@router.get("/history/{record_id}", response_model=HistoryItem)
def history_item(
    record_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> HistoryItem:
    record = db.get(DetectionRecord, record_id)
    if record is None or record.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Record not found")
    return HistoryItem.model_validate(record)
