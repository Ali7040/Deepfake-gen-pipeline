"""Video deepfake detection via spatial frame aggregation + majority voting.

Extracts evenly spaced frames, detects the most prominent face per frame (on a
resolution-capped copy), then classifies ALL collected faces in a single batched
forward pass before voting on the final verdict.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.config import settings
from app.services.ml_engine import downscale, get_engine


def detect_video(video_path: Path) -> dict:
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    engine = get_engine()
    ts = int(time.time() * 1000)
    t0 = time.perf_counter()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return {"error": "Video has no readable frames."}

    n = min(settings.video_frames, total_frames)
    frame_indices = np.linspace(0, total_frames - 1, n, dtype=int)

    # Pass 1: seek frames, detect the most prominent face, collect crops.
    collected = []  # (frame_idx, frame_bgr, (x1,y1,x2,y2), crop_pil)
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_w, img_h = pil.size

        small, scale = downscale(pil, settings.detect_max_side)
        try:
            boxes, probs = engine.mtcnn.detect(small)
        except Exception:
            continue
        if boxes is None or len(boxes) == 0 or probs[0] is None or probs[0] < settings.face_min_prob:
            continue

        # Most prominent = largest-area box (not merely boxes[0]).
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        box = boxes[int(np.argmax(areas))]

        x1, y1, x2, y2 = (int(c / scale) for c in box)  # back to full res
        w, h = x2 - x1, y2 - y1
        mx, my = int(w * 0.25), int(h * 0.25)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(img_w, x2 + mx), min(img_h, y2 + my)

        collected.append((int(idx), frame, (x1, y1, x2, y2), pil.crop((x1, y1, x2, y2))))

    cap.release()
    t_detect = time.perf_counter()

    if not collected:
        return {"error": "No clear faces detected in the video to analyze."}

    # Pass 2: one batched classification for every collected face.
    fake_probs = engine.classify_batch(engine.video, [c[3] for c in collected])
    t_classify = time.perf_counter()

    frames = []
    fake_count = 0
    for (frame_idx, frame, (x1, y1, x2, y2), _crop), prob_fake in zip(collected, fake_probs):
        label = "fake" if prob_fake >= settings.fake_threshold else "real"
        prob_real = 1.0 - prob_fake
        if label == "fake":
            fake_count += 1

        evidence_name = f"vid_{ts}_frame_{frame_idx}.png"
        color = (0, 255, 0) if label == "real" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, label.upper(), (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imwrite(str(settings.processed_dir / evidence_name), frame)

        frames.append(
            {
                "frame_number": frame_idx,
                "label": label,
                "real_confidence": round(prob_real * 100),
                "fake_confidence": round(prob_fake * 100),
                "evidence_url": f"/media/processed/{evidence_name}",
            }
        )

    fake_ratio = fake_count / len(frames)
    overall = "fake" if fake_ratio >= settings.video_fake_ratio else "real"

    return {
        "prediction": overall,
        "confidence": round(fake_ratio * 100),
        "total_analyzed_frames": len(frames),
        "fake_frames_detected": fake_count,
        "frames": frames,
        "timing_ms": {
            "detect": round((t_detect - t0) * 1000),
            "classify": round((t_classify - t_detect) * 1000),
            "total": round((time.perf_counter() - t0) * 1000),
        },
    }
