#!/usr/bin/env python3
"""
DeepTrace FastAPI Backend v2.0
- Async REST API for React dashboard
- WebSocket live progress streaming
- Background task execution (non-blocking inference)
- CORS enabled for local React dev server
- OpenAPI docs at /docs
"""
from __future__ import annotations

import os
import sys
import json
import time
import uuid
import base64
import asyncio
import logging
import threading
import traceback
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any

os.environ.setdefault('OMP_NUM_THREADS', '4')

import cv2
import numpy as np
from fastapi import (
    FastAPI, UploadFile, File, Form, HTTPException,
    WebSocket, WebSocketDisconnect, BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Make sibling `deeptrace/` importable when running from project root or api/
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
sys.path.insert(0, str(_PROJECT_ROOT))

# Reuse battle-tested helpers from simple_app.py rather than duplicating them.
# Importing the module does NOT start Flask (guarded by __main__).
from simple_app import (
    initialize_deeptrace, warmup_models, allowed_file,
    read_frame, detect_faces_in_frame, swap_frame,
    process_image_swap, _crop_face, _frame_to_b64, _b64_to_frame,
    _webcam_lock, _webcam_state, _job_progress,
)
from deeptrace import state_manager
from werkzeug.utils import secure_filename

from .schemas import (
    DetectFacesResponse, FaceCrop, SwapResponse,
    WebcamFrameRequest, WebcamFrameResponse, WebcamSourceResponse,
    ProgressResponse, ActiveJobsResponse, HealthResponse,
)
from .optimization import apply_optimizations

logger = logging.getLogger("deeptrace.api")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

UPLOAD_FOLDER = (_PROJECT_ROOT / 'deeptrace' / 'uploads').resolve()
OUTPUT_FOLDER = (_PROJECT_ROOT / 'outputs').resolve()
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

app = FastAPI(
    title="DeepTrace API",
    version="2.0.0",
    description="Async face-swap pipeline with live progress over WebSockets.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production: ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup: init models on app boot ─────────────────────────────────────────
_INIT_OK = False
_WARMUP_DONE = False
_OPT_SUMMARY: Dict[str, Any] = {}


@app.on_event("startup")
def _startup() -> None:
    global _INIT_OK, _WARMUP_DONE, _OPT_SUMMARY
    _INIT_OK = initialize_deeptrace()
    if _INIT_OK:
        try:
            _OPT_SUMMARY = apply_optimizations()
        except Exception as e:
            logger.warning("Optimization step skipped: %s", e)
        threading.Thread(target=_warmup_then_flag, daemon=True).start()
        logger.info("DeepTrace state initialised — warmup running in background")
    else:
        logger.error("DeepTrace failed to initialise")


def _warmup_then_flag() -> None:
    global _WARMUP_DONE
    warmup_models()
    _WARMUP_DONE = True


# ── Upload helper ────────────────────────────────────────────────────────────
async def _save_upload(file: UploadFile, dest_dir: Path, prefix: str) -> Path:
    if not file.filename or not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")
    ts = int(time.time() * 1000)
    safe = secure_filename(f"{prefix}_{ts}_{file.filename}")
    out = dest_dir / safe
    size = 0
    with out.open("wb") as f:
        while chunk := await file.read(1 << 20):  # 1 MB chunks
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File exceeds 100 MB limit")
            f.write(chunk)
    return out


# ── Health / meta ────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _INIT_OK else "degraded",
        init_ok=_INIT_OK,
        warmup_done=_WARMUP_DONE,
        version="2.0.0",
    )


@app.get("/api/optimization")
def optimization_info() -> Dict[str, Any]:
    """Report which providers + thread count are active."""
    return _OPT_SUMMARY or {"status": "not initialized"}


# ── Face detection ───────────────────────────────────────────────────────────
@app.post("/api/detect-faces", response_model=DetectFacesResponse)
async def api_detect_faces(image: UploadFile = File(...)) -> DetectFacesResponse:
    path = await _save_upload(image, UPLOAD_FOLDER, "detect")

    frame = read_frame(str(path))
    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot read image")

    faces = detect_faces_in_frame(frame)
    crops: List[FaceCrop] = []
    for face in faces:
        crop = _crop_face(frame, face)
        crops.append(FaceCrop(
            b64=_frame_to_b64(crop),
            score=float(face.score_set.get('detector', 0)),
            bbox=[float(v) for v in face.bounding_box],
        ))

    return DetectFacesResponse(
        success=True, count=len(faces), faces=crops, image_path=path.name,
    )


# ── Face swap (async) ────────────────────────────────────────────────────────
# Async jobs are fire-and-forget from the request/response cycle (the caller
# only gets a job_id back) — the final result would otherwise be discarded
# once the background task returns, leaving no way to retrieve output info
# after polling /api/progress/{job_id} reports "done". Stash it here instead.
_job_results: Dict[str, Dict[str, Any]] = {}


def _run_swap_blocking(src_path: str, tgt_path: str, out_path: str,
                       face_indices, enhance: bool, pitch: float,
                       job_id: str, detect_interval: int, max_side: int) -> Dict[str, Any]:
    try:
        result = process_image_swap(
            src_path, tgt_path, out_path,
            face_indices=face_indices,
            enhance=enhance,
            pitch_semitones=pitch,
            job_id=job_id,
            detect_interval=detect_interval,
            max_side=max_side,
        )
        if result.get('success'):
            result['output_filename'] = Path(out_path).name
            result['output_url'] = f"/output/{Path(out_path).name}"
            if result.get('output_type') == 'image' and Path(out_path).is_file():
                with open(out_path, 'rb') as f:
                    result['preview_b64'] = base64.b64encode(f.read()).decode()
        if job_id:
            _job_results[job_id] = result
        return result
    except Exception:
        logger.error("swap job %s crashed:\n%s", job_id, traceback.format_exc())
        failure = {'success': False, 'error': 'Internal error during swap'}
        if job_id:
            _job_results[job_id] = failure
        return failure


@app.get("/api/result/{job_id}", response_model=SwapResponse)
async def api_result(job_id: str) -> SwapResponse:
    """Fetch the final result of an async swap job once /api/progress reports
    status == "done" (or "error"). 404 while still processing/unknown."""
    result = _job_results.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not ready or unknown job_id")
    return SwapResponse(job_id=job_id, **result)


@app.post("/api/swap", response_model=SwapResponse)
async def api_swap(
    background_tasks: BackgroundTasks,
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    face_indices: str = Form(""),
    enhance: str = Form("1"),
    pitch_semitones: float = Form(0.0),
    detect_interval: int = Form(5),
    max_side: int = Form(720),
    async_mode: bool = Form(False),
) -> SwapResponse:
    """
    Face swap endpoint.
    - When `async_mode=true`, returns immediately with a job_id; track progress
      via WebSocket /ws/job/{job_id} or GET /api/progress/{job_id}.
    - When `async_mode=false` (default), runs in threadpool and returns full result.
    """
    src_path = await _save_upload(source, UPLOAD_FOLDER, "source")
    tgt_path = await _save_upload(target, UPLOAD_FOLDER, "target")

    parsed_indices = None
    if face_indices.strip():
        try:
            parsed_indices = list({int(i) for i in json.loads(face_indices)})
        except Exception:
            parsed_indices = None

    do_enhance = enhance == "1" or enhance.lower() == "true"
    job_id = uuid.uuid4().hex

    tgt_ext = Path(target.filename or "").suffix.lower()
    if tgt_ext in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
        ext = tgt_ext
    elif tgt_ext in {'.mp4', '.avi', '.mov', '.mkv'}:
        ext = '.mp4'
    else:
        ext = '.jpg'
    out_name = f"output_{int(time.time())}_{job_id[:8]}{ext}"
    out_path = OUTPUT_FOLDER / out_name

    _job_progress[job_id] = {
        'total': 0, 'done': 0, 'status': 'queued', 'eta_seconds': 0, 'fps_proc': 0.0,
    }

    if async_mode:
        # Fire-and-forget; client polls /ws or /api/progress
        background_tasks.add_task(
            _run_swap_blocking, str(src_path), str(tgt_path), str(out_path),
            parsed_indices, do_enhance, pitch_semitones, job_id,
            detect_interval, max_side,
        )
        return SwapResponse(success=True, job_id=job_id)

    # Sync: run inference off the event loop so other requests stay responsive
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        _run_swap_blocking,
        str(src_path), str(tgt_path), str(out_path),
        parsed_indices, do_enhance, pitch_semitones, job_id,
        detect_interval, max_side,
    )
    if not result.get('success'):
        return SwapResponse(success=False, job_id=job_id, error=result.get('error'))
    return SwapResponse(job_id=job_id, **result)


# ── Webcam ───────────────────────────────────────────────────────────────────
@app.post("/api/webcam/set-source", response_model=WebcamSourceResponse)
async def webcam_set_source(source: UploadFile = File(...)) -> WebcamSourceResponse:
    path = await _save_upload(source, UPLOAD_FOLDER, "webcam_src")

    frame = read_frame(str(path))
    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot read source image")

    faces = detect_faces_in_frame(frame)
    if not faces:
        return WebcamSourceResponse(
            success=False,
            error="No face detected in source. Use a clear front-facing portrait.",
        )

    state_manager.set_item('source_paths', [str(path)])
    with _webcam_lock:
        _webcam_state['source_face'] = faces[0]
        _webcam_state['source_path'] = str(path)
        _webcam_state['last_result'] = None

    crop = _crop_face(frame, faces[0])
    return WebcamSourceResponse(
        success=True,
        preview_b64=_frame_to_b64(crop),
        score=float(faces[0].score_set.get('detector', 0)),
    )


@app.post("/api/webcam/process-frame", response_model=WebcamFrameResponse)
async def webcam_process_frame(req: WebcamFrameRequest) -> WebcamFrameResponse:
    with _webcam_lock:
        source_face = _webcam_state.get('source_face')

    if source_face is None:
        return WebcamFrameResponse(success=False, error="No source face set — upload one first")

    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    frame = await loop.run_in_executor(None, _b64_to_frame, req.frame_b64)
    if frame is None:
        return WebcamFrameResponse(success=False, error="Cannot decode frame")

    def _do_swap():
        tfaces = detect_faces_in_frame(frame)
        out = frame
        if tfaces:
            out = swap_frame(source_face, tfaces, frame, enhance=req.enhance)
        return out, len(tfaces)

    result_bgr, n_faces = await loop.run_in_executor(None, _do_swap)
    elapsed = time.perf_counter() - t0
    fps = round(1.0 / max(elapsed, 0.001), 1)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return WebcamFrameResponse(
        success=True,
        result_b64=_frame_to_b64(result_rgb, quality=70),
        fps=fps,
        faces_found=n_faces,
    )


# ── Output file serving ──────────────────────────────────────────────────────
@app.get("/output/{filename:path}")
def get_output(filename: str):
    safe = Path(filename).name  # strip any path components
    full = OUTPUT_FOLDER / safe
    if not full.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(full))


# ── Job progress (REST polling fallback) ─────────────────────────────────────
@app.get("/api/progress/{job_id}", response_model=ProgressResponse)
def api_progress(job_id: str) -> ProgressResponse:
    p = _job_progress.get(job_id)
    if p is None:
        return ProgressResponse(found=False)
    return ProgressResponse(found=True, **p)


@app.get("/api/active-jobs", response_model=ActiveJobsResponse)
def api_active_jobs() -> ActiveJobsResponse:
    active = [
        {'job_id': jid, **info}
        for jid, info in _job_progress.items()
        if info.get('status') in ('processing', 'encoding', 'uploaded', 'queued')
    ]
    return ActiveJobsResponse(jobs=active)


# ── WebSocket: live progress stream ──────────────────────────────────────────
@app.websocket("/ws/job/{job_id}")
async def ws_job_progress(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        last_snapshot: Optional[str] = None
        # Stream until job done or client disconnects
        for _ in range(60 * 60 * 4):  # ~4h hard cap @ 0.25s tick
            p = _job_progress.get(job_id)
            if p is None:
                await websocket.send_json({'found': False, 'job_id': job_id})
                await asyncio.sleep(0.5)
                continue

            payload = {'found': True, 'job_id': job_id, **p}
            snapshot = json.dumps(payload, sort_keys=True)
            if snapshot != last_snapshot:
                await websocket.send_json(payload)
                last_snapshot = snapshot

            if p.get('status') in ('done', 'error'):
                break
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        logger.info("WS disconnect for job %s", job_id)
    except Exception:
        logger.error("WS error for job %s:\n%s", job_id, traceback.format_exc())
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ── Backward-compatible /swap alias ──────────────────────────────────────────
@app.post("/swap", response_model=SwapResponse, include_in_schema=False)
async def swap_compat(
    background_tasks: BackgroundTasks,
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    face_indices: str = Form(""),
    enhance: str = Form("1"),
    pitch_semitones: float = Form(0.0),
    detect_interval: int = Form(5),
    max_side: int = Form(720),
) -> SwapResponse:
    return await api_swap(
        background_tasks, source, target, face_indices, enhance,
        pitch_semitones, detect_interval, max_side, async_mode=False,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,  # single worker — models are loaded in-process
        log_level="info",
    )
