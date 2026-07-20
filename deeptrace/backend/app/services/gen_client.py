"""Async HTTP client to the DeepTrace face-swap engine (existing FastAPI).

The swap pipeline (onnxruntime + the deeptrace package) runs as its own process
so its heavy, distinct dependency stack stays isolated from the detection stack.
This module is the single point that talks to it.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.config import settings

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=settings.gen_base_url,
            timeout=httpx.Timeout(settings.gen_timeout, connect=10.0),
        )
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def fastapi_health_async() -> dict[str, Any]:
    """Best-effort reachability probe of the swap engine."""
    try:
        resp = await get_client().get("/api/health", timeout=5.0)
        return {"reachable": True, "status": resp.status_code, "body": resp.json()}
    except Exception as exc:
        return {"reachable": False, "error": str(exc), "base_url": settings.gen_base_url}


def fastapi_health() -> dict[str, Any]:
    """Sync probe (used from the aggregate /api/health)."""
    try:
        resp = httpx.get(f"{settings.gen_base_url}/api/health", timeout=5.0)
        return {"reachable": True, "status": resp.status_code, "body": resp.json()}
    except Exception as exc:
        return {"reachable": False, "error": str(exc), "base_url": settings.gen_base_url}
