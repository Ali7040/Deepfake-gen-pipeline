"""Image deepfake detection.

Pipeline: MTCNN face detection (on a resolution-capped copy for speed) → crop
each face from the FULL-resolution image → single batched ViT/SigLIP forward pass
→ annotate. Returns a plain dict; heavy libs are imported lazily.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.config import settings
from app.services.ml_engine import downscale, get_engine


def detect_image(image_path: Path) -> dict:
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    engine = get_engine()
    ts = int(time.time() * 1000)
    t0 = time.perf_counter()

    try:
        image = PILImage.open(image_path).convert("RGB")
    except Exception:
        return {"error": "Could not read image file."}
    img_w, img_h = image.size

    # Detect on a downscaled copy (MTCNN cost ∝ pixels); map boxes back to full res.
    small, scale = downscale(image, settings.detect_max_side)
    try:
        boxes, probs = engine.mtcnn.detect(small)
    except Exception:
        return {"error": "No human faces detected."}
    if boxes is None or len(boxes) == 0:
        return {"error": "No human faces detected."}
    t_detect = time.perf_counter()

    # Full-res BGR for annotation (reuse the already-decoded PIL, no second read).
    cv_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    # Pass 1: collect valid face crops (from full-res) + their boxes.
    crops, meta = [], []
    for i, box in enumerate(boxes):
        if probs[i] is None or probs[i] < settings.face_min_prob:
            continue
        x1, y1, x2, y2 = (int(c / scale) for c in box)  # back to full res
        w, h = x2 - x1, y2 - y1
        mx, my = int(w * 0.20), int(h * 0.20)  # portrait margin
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(img_w, x2 + mx), min(img_h, y2 + my)
        crops.append(image.crop((x1, y1, x2, y2)))
        meta.append((i, x1, y1, x2, y2))

    if not crops:
        return {"error": "No valid faces verified (low detector confidence)."}

    # Pass 2: one batched classification for all faces.
    fake_probs = engine.classify_batch(engine.image, crops)
    t_classify = time.perf_counter()

    faces = []
    for (idx, x1, y1, x2, y2), cropped, prob_fake in zip(meta, crops, fake_probs):
        label = "fake" if prob_fake >= settings.fake_threshold else "real"
        prob_real = 1.0 - prob_fake

        crop_name = f"crop_{ts}_{idx}.png"
        cropped.save(settings.processed_dir / crop_name)

        faces.append(
            {
                "label": label,
                "real_confidence": round(prob_real * 100),
                "fake_confidence": round(prob_fake * 100),
                "threat": "Synthetic Media Detected" if label == "fake" else "Authentic Media",
                "image_url": f"/media/processed/{crop_name}",
            }
        )
        color = (0, 255, 0) if label == "real" else (0, 0, 255)
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            cv_image,
            f"{label.upper()} ({max(prob_real, prob_fake) * 100:.0f}%)",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    processed_name = f"processed_{ts}.png"
    cv2.imwrite(str(settings.processed_dir / processed_name), cv_image)

    any_fake = any(f["label"] == "fake" for f in faces)
    overall = "fake" if any_fake else "real"
    if any_fake:
        confidence = max(f["fake_confidence"] for f in faces if f["label"] == "fake")
    else:
        confidence = max(f["real_confidence"] for f in faces)

    return {
        "prediction": overall,
        "confidence": confidence,
        "face_count": len(faces),
        "faces": faces,
        "processed_image_url": f"/media/processed/{processed_name}",
        "timing_ms": {
            "detect": round((t_detect - t0) * 1000),
            "classify": round((t_classify - t_detect) * 1000),
            "total": round((time.perf_counter() - t0) * 1000),
        },
    }
