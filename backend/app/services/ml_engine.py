"""Lazy, process-wide ML model singletons for deepfake detection.

Models are heavy (torch + transformers + facenet) and their imports are slow,
so nothing here is imported at module load. The first detection request pays
the one-time load cost; subsequent requests reuse the in-memory models.

Thread-safe: loads are guarded by a lock so concurrent first-requests don't
double-load. Inference itself is CPU/GPU bound and should be dispatched off the
event loop (the routers use `run_in_executor`).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from app.config import settings

logger = logging.getLogger("deeptrace.ml")

_lock = threading.Lock()
_engine: "MLEngine | None" = None


class DetectionUnavailable(RuntimeError):
    """Raised when the detection ML stack cannot be loaded (missing deps)."""


@dataclass
class _Bundle:
    processor: Any
    model: Any


def downscale(pil_image, max_side: int):
    """Return (possibly-downscaled image, scale). Detect on the smaller image,
    then divide box coords by `scale` to map back to full resolution."""
    if not max_side:
        return pil_image, 1.0
    w, h = pil_image.size
    longest = max(w, h)
    if longest <= max_side:
        return pil_image, 1.0
    scale = max_side / longest
    return pil_image.resize((max(1, int(w * scale)), max(1, int(h * scale)))), scale


class MLEngine:
    """Holds the face detector and the image/video classifiers."""

    def __init__(self) -> None:
        import torch  # local import: keep torch out of app boot path
        from facenet_pytorch import MTCNN
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pin CPU threads if configured (avoids thread oversubscription).
        if settings.torch_num_threads > 0 and self.device.type == "cpu":
            torch.set_num_threads(settings.torch_num_threads)
            logger.info("torch CPU threads pinned to %d", settings.torch_num_threads)

        logger.info("Initialising detection engine on %s", self.device)

        self.mtcnn = MTCNN(keep_all=True, device=self.device)

        # Use Auto* loaders (not hardcoded ViT): the checkpoints differ in
        # architecture — e.g. the default image model is SigLIP, not ViT — so a
        # fixed class silently random-initialises mismatched weights.
        def _load(model_id: str) -> _Bundle:
            return _Bundle(
                processor=AutoImageProcessor.from_pretrained(model_id, use_fast=True),
                model=AutoModelForImageClassification.from_pretrained(model_id)
                .to(self.device)
                .eval(),
            )

        logger.info("Loading image classifier: %s", settings.image_model)
        self.image = _load(settings.image_model)

        if settings.video_model == settings.image_model:
            self.video = self.image  # reuse to save memory
        else:
            logger.info("Loading video classifier: %s", settings.video_model)
            self.video = _load(settings.video_model)
        logger.info("Detection engine ready.")

    @staticmethod
    def _fake_index(id2label: dict) -> int | None:
        for idx, label in id2label.items():
            if "fake" in label.lower():
                return int(idx)
        return None

    def classify_batch(self, bundle: _Bundle, pil_images: list) -> list[float]:
        """Return P(fake) in [0, 1] for a list of cropped RGB face images in a
        single batched forward pass (much cheaper than one call per face)."""
        if not pil_images:
            return []
        import torch.nn.functional as F

        inputs = bundle.processor(images=pil_images, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            logits = bundle.model(**inputs).logits
        probs = F.softmax(logits, dim=1)  # [N, C]
        id2label = bundle.model.config.id2label
        fake_idx = self._fake_index(id2label)

        out: list[float] = []
        for row in probs:
            if fake_idx is not None:
                out.append(float(row[fake_idx].item()))
            else:  # model with no explicit "fake" label -> argmax semantics
                top = int(row.argmax().item())
                out.append(0.0 if "real" in id2label[top].lower() else float(row[top].item()))
        return out

    def fake_probability(self, bundle: _Bundle, pil_image) -> float:
        """Convenience single-image wrapper over classify_batch."""
        return self.classify_batch(bundle, [pil_image])[0]

    def warmup(self) -> None:
        """Run one tiny inference so the first real request isn't cold."""
        from PIL import Image

        dummy = Image.new("RGB", (224, 224))
        try:
            self.classify_batch(self.image, [dummy])
            if self.video is not self.image:
                self.classify_batch(self.video, [dummy])
            logger.info("Detection engine warmed up.")
        except Exception as exc:  # non-fatal
            logger.warning("Warmup skipped: %s", exc)


def get_engine() -> MLEngine:
    """Return the shared engine, loading it on first use (thread-safe)."""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                try:
                    _engine = MLEngine()
                except ImportError as exc:
                    raise DetectionUnavailable(
                        "Detection ML stack is not installed. Run "
                        "`pip install -r requirements-ml.txt` in the backend "
                        f"environment. (missing: {exc.name})"
                    ) from exc
    return _engine


def engine_status() -> dict[str, Any]:
    """Non-loading status probe for health checks."""
    return {
        "loaded": _engine is not None,
        "image_model": settings.image_model,
        "video_model": settings.video_model,
    }
