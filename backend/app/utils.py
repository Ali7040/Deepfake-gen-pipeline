"""Small shared helpers (safe upload saving)."""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from app.config import settings

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif", ".flv", ".wmv", ".3gp"}

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_stem(name: str) -> str:
    return _SAFE.sub("_", Path(name).stem)[:60] or "file"


async def save_upload(
    file: UploadFile, prefix: str, allowed_exts: set[str]
) -> Path:
    """Stream an upload to the uploads dir, enforcing extension + size limits."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported file type '{ext or '?'}'. Allowed: {sorted(allowed_exts)}",
        )

    dest = settings.upload_dir / f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}_{_safe_stem(file.filename or '')}{ext}"
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1 << 20):  # 1 MB
            size += len(chunk)
            if size > settings.max_upload_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    f"File exceeds {settings.max_upload_bytes // (1024 * 1024)} MB limit",
                )
            out.write(chunk)
    return dest
