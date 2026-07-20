"""Generation endpoints — thin proxy to the DeepTrace face-swap engine.

The frontend only ever talks to this backend; these routes forward to the swap
FastAPI (default http://127.0.0.1:8000) so the swap pipeline stays a separate,
independently-deployable process. Swap requires a valid access token; read-only
probes are open.

Live progress: for a running job the frontend can either poll
GET /api/generation/progress/{job_id}, or connect its WebSocket directly to the
swap engine at ws://<gen_base_url>/ws/job/{job_id} (URL exposed via /config).
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from app.config import settings
from app.deps import get_current_user
from app.models import User
from app.services.gen_client import fastapi_health_async, get_client

router = APIRouter(prefix="/generation", tags=["generation"])


def _bad_gateway(exc: Exception) -> HTTPException:
    return HTTPException(
        status.HTTP_502_BAD_GATEWAY,
        f"Face-swap engine unreachable at {settings.gen_base_url}: {exc}",
    )


async def _forward_files(
    path: str, files: list[tuple[str, UploadFile]], data: dict[str, Any]
) -> Response:
    """Re-post uploaded files + form data to the swap engine, mirror the reply."""
    multipart = [
        (field, (f.filename, await f.read(), f.content_type or "application/octet-stream"))
        for field, f in files
    ]
    try:
        resp = await get_client().post(path, files=multipart, data=data)
    except httpx.HTTPError as exc:
        raise _bad_gateway(exc)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


@router.get("/config")
async def gen_config(user: User = Depends(get_current_user)) -> dict[str, Any]:
    """Expose the swap engine base + WebSocket URL for direct progress streaming."""
    ws = settings.gen_base_url.replace("http://", "ws://").replace("https://", "wss://")
    return {
        "base_url": settings.gen_base_url,
        "progress_ws_template": f"{ws}/ws/job/{{job_id}}",
    }


@router.get("/health")
async def gen_health() -> dict[str, Any]:
    return await fastapi_health_async()


@router.post("/detect-faces")
async def gen_detect_faces(
    image: UploadFile = File(...),
    user: User = Depends(get_current_user),
) -> Response:
    return await _forward_files("/api/detect-faces", [("image", image)], {})


@router.post("/swap")
async def gen_swap(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    face_indices: str = Form(""),
    enhance: str = Form("1"),
    pitch_semitones: float = Form(0.0),
    detect_interval: int = Form(5),
    max_side: int = Form(720),
    async_mode: bool = Form(False),
    user: User = Depends(get_current_user),
) -> Response:
    """Proxy a face-swap request. Set async_mode=true to get a job_id back and
    track progress via /progress/{job_id} or the swap engine WebSocket."""
    data = {
        "face_indices": face_indices,
        "enhance": enhance,
        "pitch_semitones": pitch_semitones,
        "detect_interval": detect_interval,
        "max_side": max_side,
        "async_mode": str(async_mode).lower(),
    }
    return await _forward_files(
        "/api/swap", [("source", source), ("target", target)], data
    )


@router.get("/progress/{job_id}")
async def gen_progress(job_id: str, user: User = Depends(get_current_user)) -> Response:
    try:
        resp = await get_client().get(f"/api/progress/{job_id}")
    except httpx.HTTPError as exc:
        raise _bad_gateway(exc)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


@router.get("/active-jobs")
async def gen_active_jobs(user: User = Depends(get_current_user)) -> Response:
    try:
        resp = await get_client().get("/api/active-jobs")
    except httpx.HTTPError as exc:
        raise _bad_gateway(exc)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


@router.get("/output/{filename}")
async def gen_output(filename: str) -> StreamingResponse:
    """Stream a rendered swap result back from the swap engine."""
    client = get_client()
    try:
        req = client.build_request("GET", f"/output/{filename}")
        resp = await client.send(req, stream=True)
    except httpx.HTTPError as exc:
        raise _bad_gateway(exc)
    if resp.status_code != 200:
        await resp.aclose()
        raise HTTPException(resp.status_code, "Output file not found")
    return StreamingResponse(
        resp.aiter_bytes(),
        media_type=resp.headers.get("content-type", "application/octet-stream"),
        background=BackgroundTask(resp.aclose),
    )
