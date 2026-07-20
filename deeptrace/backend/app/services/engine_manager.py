"""Optionally auto-start the generation engine as a child process.

Enabled with `AUTO_START_GEN_ENGINE=true`. Off by default because the normal dev
flow (start.bat) runs the engine in its own window, and a child process fights
uvicorn's --reload. When enabled it only spawns if the engine port is free and
the engine is actually installed (deeptrace/ present).

Caveat: the child is cleaned up on graceful shutdown (Ctrl+C). A force-kill of
the backend can orphan it — use start.bat if you want fully independent process
lifecycles.
"""

from __future__ import annotations

import logging
import socket
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from app.config import BASE_DIR, settings

logger = logging.getLogger("deeptrace.engine")


def _venv_python(venv_dir: Path) -> Path:
    win = venv_dir / "Scripts" / "python.exe"
    return win if win.exists() else venv_dir / "bin" / "python"


def _port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def start_engine_if_configured() -> subprocess.Popen | None:
    if not settings.auto_start_gen_engine:
        return None

    u = urlparse(settings.gen_base_url)
    host, port = (u.hostname or "127.0.0.1"), (u.port or 8000)

    if _port_open(host, port):
        logger.info("Gen engine already running on %s:%s — not spawning.", host, port)
        return None

    # Consolidated repo: the engine (api/main.py + venv/) sits at the repo root,
    # which is the parent of backend/.
    gen_dir = Path(settings.gen_engine_dir) if settings.gen_engine_dir else BASE_DIR.parent
    py = _venv_python(gen_dir / "venv")
    if not py.exists() or not (gen_dir / "api" / "main.py").exists():
        logger.warning(
            "Gen engine not installed at %s — generation disabled. Run setup.bat.", gen_dir
        )
        return None

    logger.info("Auto-starting gen engine from %s on port %s ...", gen_dir, port)
    return subprocess.Popen(
        [str(py), "-m", "uvicorn", "api.main:app", "--port", str(port)],
        cwd=str(gen_dir),
    )


def stop_engine(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    logger.info("Stopping auto-started gen engine ...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
