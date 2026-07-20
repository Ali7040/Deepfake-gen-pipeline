"""FastAPI application entrypoint for the DeepTrace unified backend."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import settings
from app.database import init_db
from app.routers import auth, detection, generation
from app.services.gen_client import close_client, fastapi_health_async
from app.services.ml_engine import DetectionUnavailable, engine_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("deeptrace.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.services.engine_manager import start_engine_if_configured, stop_engine

    settings.ensure_dirs()
    init_db()
    engine_proc = start_engine_if_configured()
    if settings.warmup_on_startup:
        import threading

        from app.services.ml_engine import get_engine

        def _warm():
            try:
                get_engine().warmup()
            except Exception as exc:
                logger.warning("Startup warmup failed: %s", exc)

        threading.Thread(target=_warm, daemon=True).start()
        logger.info("DeepTrace backend ready (warming up detection models in background).")
    else:
        logger.info("DeepTrace backend ready (detection models load lazily on first use).")
    yield
    stop_engine(engine_proc)
    await close_client()


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description=(
        "Unified DeepTrace API: native deepfake **detection** plus a proxy to the "
        "face-swap **generation** engine. JWT-authenticated."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.cors_allow_all else settings.cors_origin_list,
    allow_credentials=not settings.cors_allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(DetectionUnavailable)
async def _detection_unavailable_handler(request: Request, exc: DetectionUnavailable):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


# Serve uploaded originals + processed evidence images.
app.mount("/media", StaticFiles(directory=str(settings.media_dir)), name="media")

# API routers (all under /api).
app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(detection.router, prefix=settings.api_prefix)
app.include_router(generation.router, prefix=settings.api_prefix)


@app.get("/api", tags=["meta"])
async def api_root() -> dict[str, Any]:
    return {
        "service": settings.app_name,
        "version": __version__,
        "docs": "/docs",
        "sections": {
            "auth": f"{settings.api_prefix}/auth",
            "detection": f"{settings.api_prefix}/detection",
            "generation": f"{settings.api_prefix}/generation",
        },
    }


@app.get("/api/health", tags=["meta"])
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "detection_engine": engine_status(),
        "generation_engine": await fastapi_health_async(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)
