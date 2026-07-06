"""Application settings, loaded from environment / .env (pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────────────
    app_name: str = "DeepTrace API"
    debug: bool = True
    api_prefix: str = "/api"

    # ── Security / JWT ─────────────────────────────────────────────────────
    secret_key: str = Field(
        default="dev-insecure-change-me", validation_alias="SECRET_KEY"
    )
    jwt_algorithm: str = "HS256"
    access_token_ttl_min: int = 60
    refresh_token_ttl_days: int = 7

    # ── CORS ───────────────────────────────────────────────────────────────
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:3000,http://127.0.0.1:3000"
    )
    cors_allow_all: bool = True  # convenient in dev; set False in prod

    # ── Storage ────────────────────────────────────────────────────────────
    database_url: str = Field(
        default=f"sqlite:///{(BASE_DIR / 'deeptrace.db').as_posix()}",
        validation_alias="DATABASE_URL",
    )
    media_dir: Path = BASE_DIR / "media"
    max_upload_bytes: int = 100 * 1024 * 1024  # 100 MB

    # ── Detection model config ─────────────────────────────────────────────
    image_model: str = "prithivMLmods/Deep-Fake-Detector-Model"
    video_model: str = "dima806/deepfake_vs_real_image_detection"
    face_min_prob: float = 0.90
    fake_threshold: float = 0.50
    video_frames: int = 10
    video_fake_ratio: float = 0.30

    # ── Detection performance ──────────────────────────────────────────────
    # Downscale so the longest side is <= this before running MTCNN (0 = off).
    # MTCNN cost scales with pixel count; face crops are still taken from the
    # full-resolution image, so accuracy is preserved.
    detect_max_side: int = 1024
    # Pin torch CPU threads (0 = leave torch's default). Avoids oversubscription.
    torch_num_threads: int = 0
    # Load + warm up the detection models in a background thread on startup so
    # the first user request isn't cold. Off by default to keep boot fast.
    warmup_on_startup: bool = False

    # ── Generation proxy (existing FastAPI swap engine) ────────────────────
    gen_base_url: str = "http://127.0.0.1:8000"
    gen_timeout: float = 600.0
    # Auto-spawn the gen engine as a subprocess when the backend boots, so a
    # single `uvicorn app.main:app` brings up both. On by default in this
    # consolidated repo. Best used WITHOUT --reload (reload restarts the engine
    # on every code change); start.bat runs it in its own window instead.
    auto_start_gen_engine: bool = True
    gen_engine_dir: str = ""  # default: repo root (parent of backend/)

    # ── Forensic report (Qwen via HF Inference API) ────────────────────────
    hf_token: str = ""
    report_model_url: str = (
        "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct"
    )

    # ── Derived helpers ────────────────────────────────────────────────────
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def upload_dir(self) -> Path:
        return self.media_dir / "uploads"

    @property
    def processed_dir(self) -> Path:
        return self.media_dir / "processed"

    def ensure_dirs(self) -> None:
        for d in (self.media_dir, self.upload_dir, self.processed_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
