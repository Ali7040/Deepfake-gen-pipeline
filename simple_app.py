#!/usr/bin/env python3
"""
DeepTrace Face Swap Web Application v2.0
- Multi-face detection and selective target swapping
- Live webcam deepfake with source face overlay
- Audio pitch-shift for video outputs
- Optimized ONNX inference pipeline
"""

import os
import base64
import time
import logging
import traceback
import subprocess
import threading
import uuid
from io import BytesIO
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '4'

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string, send_file, send_from_directory, Response
from werkzeug.utils import secure_filename

# Import deeptrace modules
from deeptrace import state_manager, face_detector, face_analyser
from deeptrace.processors.modules.face_swapper import core as face_swapper
import deeptrace.processors.modules.face_enhancer.core as face_enhancer
from deeptrace.vision import read_static_image, write_image
from deeptrace.filesystem import is_image, is_video

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deeptrace_app.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
# Compute paths: script is in deeptrace folder, but outputs/uploads are at project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB
app.config['UPLOAD_FOLDER'] = os.path.abspath(os.path.join(_PROJECT_ROOT, 'deeptrace', 'uploads'))
app.config['OUTPUT_FOLDER'] = os.path.abspath(os.path.join(_PROJECT_ROOT, 'outputs'))

Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'mp4', 'avi', 'mov', 'mkv'}

_INIT_OK = False       # set after initialize_deeptrace()
_WARMUP_DONE = False   # set after model warm-up
_job_progress: dict = {}  # job_id -> {total, done, status}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── State initialisation ──────────────────────────────────────────────────────
def initialize_deeptrace() -> bool:
    """Initialize DeepTrace state.  Lower thresholds so marginal faces are caught."""
    try:
        state_manager.init_item('download_providers', ['github', 'huggingface'])
        state_manager.init_item('download_scope', 'full')
        state_manager.init_item('log_level', 'info')

        # Execution
        state_manager.init_item('execution_providers', ['CPUExecutionProvider'])
        state_manager.init_item('execution_device_ids', [0])
        state_manager.init_item('execution_thread_count', 4)

        # Face detector  ← BUG FIX: was 0.3, raised false-negative rate
        state_manager.init_item('face_detector_model', 'yolo_face')
        state_manager.init_item('face_detector_size', '640x640')
        state_manager.init_item('face_detector_score', 0.15)   # lowered from 0.3
        state_manager.init_item('face_detector_angles', [0])
        state_manager.init_item('face_detector_margin', [0, 0, 0, 0])

        # Face swapper
        state_manager.init_item('face_swapper_model', 'inswapper_128_fp16')
        state_manager.init_item('face_swapper_pixel_boost', '256x256')  # was 128x128 → sharper swap region
        state_manager.init_item('face_swapper_weight', 1.0)             # was 0.5 → full swap, no ghosting

        # Face enhancer
        state_manager.init_item('face_enhancer_model', 'gfpgan_1.4')
        state_manager.init_item('face_enhancer_blend', 100)  # was 80 → full GFPGAN output, no original leak
        state_manager.init_item('face_enhancer_weight', 1.0)

        # Recogniser / landmarker
        state_manager.init_item('face_recognizer_model', 'arcface_inswapper')
        state_manager.init_item('face_landmarker_model', '2dfan4')
        state_manager.init_item('face_landmarker_score', 0.3)

        # Masks — softer blur + padding captures full face boundary cleanly
        state_manager.init_item('face_mask_types', ['box'])
        state_manager.init_item('face_mask_blur', 0.6)              # softer inswapper edge feathering
        state_manager.init_item('face_mask_padding', [0, 20, 30, 20])  # top/right/bottom/left — more bottom for jaw
        state_manager.init_item('face_mask_areas', [])
        state_manager.init_item('face_mask_regions', [])
        state_manager.init_item('face_occluder_model', 'xseg_1')
        state_manager.init_item('face_parser_model', 'bisenet_resnet_34')

        # Selector
        state_manager.init_item('face_selector_mode', 'many')
        state_manager.init_item('face_selector_order', 'large-small')
        state_manager.init_item('reference_face_distance', 0.6)

        # Output / misc
        state_manager.init_item('output_path', app.config['OUTPUT_FOLDER'])
        state_manager.init_item('temp_path', '.temp')
        state_manager.init_item('output_image_quality', 90)
        state_manager.init_item('video_memory_strategy', 'moderate')
        state_manager.init_item('source_paths', [])

        logger.info('DeepTrace state initialised')
        return True
    except Exception:
        logger.error('Failed to initialise DeepTrace:\n' + traceback.format_exc())
        return False


def warmup_models() -> None:
    """Run a small synthetic forward pass so inference pools are pre-loaded."""
    global _WARMUP_DONE
    try:
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        face_analyser.get_many_faces([dummy])
        logger.info('Model warm-up complete')
    except Exception as e:
        logger.warning(f'Warm-up skipped ({e})')
    _WARMUP_DONE = True


# ── Helpers ───────────────────────────────────────────────────────────────────
def _crop_face(frame: np.ndarray, face) -> np.ndarray:
    """Return a tight crop of `face` from `frame` (RGB, 128×128)."""
    x1, y1, x2, y2 = [int(v) for v in face.bounding_box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
    crop_rgb = cv2.cvtColor(cv2.resize(crop, (128, 128)), cv2.COLOR_BGR2RGB)
    return crop_rgb


def _frame_to_b64(frame_rgb: np.ndarray, quality: int = 75) -> str:
    """Encode an RGB numpy array to a JPEG base64 string."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def _b64_to_frame(b64str: str) -> np.ndarray:
    """Decode base64 JPEG to a BGR numpy array."""
    raw = base64.b64decode(b64str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=3)
        return True
    except Exception:
        return False


def _has_audio_stream(path: str) -> bool:
    """Return True if the file contains at least one audio stream."""
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
             '-show_entries', 'stream=codec_type',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, timeout=10
        )
        return r.returncode == 0 and b'audio' in r.stdout
    except Exception:
        return False


def _atempo_chain(rate: float) -> str:
    """Build an atempo filter chain that stays within ffmpeg's 0.5-2.0 limit."""
    if 0.5 <= rate <= 2.0:
        return f'atempo={rate:.6f}'
    # Split into two equal stages
    stage = rate ** 0.5
    return f'atempo={stage:.6f},atempo={stage:.6f}'


def _mux_video_audio(raw_video: str, audio_src: str, output: str,
                     pitch_semitones: float = 0.0) -> bool:
    """
    Re-encode raw_video to H.264, mux audio from audio_src.
    Returns True on success.
    """
    has_audio = _has_audio_stream(audio_src)
    logger.info(f'Audio mux: {"found" if has_audio else "no audio"} in target')

    cmd = ['ffmpeg', '-y', '-i', raw_video]
    if has_audio:
        cmd += ['-i', audio_src]

    # Video: H.264, fast, web-optimised
    cmd += ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart']

    if has_audio:
        cmd += ['-map', '0:v:0', '-map', '1:a:0']
        cmd += ['-c:a', 'aac', '-b:a', '128k', '-ar', '44100']
        if pitch_semitones != 0.0:
            rate = 2 ** (pitch_semitones / 12)
            af = f'asetrate=44100*{rate:.6f},{_atempo_chain(rate)},aresample=44100'
            cmd += ['-af', af]
    else:
        cmd += ['-map', '0:v:0']

    cmd.append(output)

    try:
        r = subprocess.run(cmd, capture_output=True, timeout=600)
        if r.returncode != 0:
            logger.error(f'ffmpeg mux failed:\n{r.stderr.decode(errors="replace")}')
        return r.returncode == 0
    except Exception as e:
        logger.error(f'ffmpeg mux exception: {e}')
        return False


# ── Professional face blending (LAB colour transfer + Laplacian pyramid) ──────
def _build_ellipse_mask(h: int, w: int, face,
                         pad_x_frac: float = 0.22,
                         pad_top_frac: float = 0.12,
                         pad_bot_frac: float = 0.38) -> np.ndarray:
    """
    Elliptical, Gaussian-feathered face mask sized to the detected bounding box.
    Extra bottom padding covers the chin/jaw area that box masks miss.
    """
    x1, y1, x2, y2 = [int(v) for v in face.bounding_box]
    fw, fh = x2 - x1, y2 - y1

    px  = max(12, int(fw * pad_x_frac))
    pt  = max(8,  int(fh * pad_top_frac))
    pb  = max(18, int(fh * pad_bot_frac))

    x1e = max(0, x1 - px);  y1e = max(0, y1 - pt)
    x2e = min(w, x2 + px);  y2e = min(h, y2 + pb)

    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = (x1e + x2e) // 2, (y1e + y2e) // 2
    rx, ry = (x2e - x1e) // 2, (y2e - y1e) // 2
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    # Feather: kernel ≥ face-width/8, always odd
    k = max(31, (min(rx, ry) // 4) * 2 + 1)
    return cv2.GaussianBlur(mask, (k, k), 0)


def _lab_colour_transfer(swapped: np.ndarray, original: np.ndarray,
                          mask: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """
    Transfer LAB colour statistics from original→swapped in the OUTER RING of
    the face mask only (boundary zone).  The face centre is untouched so the
    source identity is preserved; only the edge pixels shift toward the target
    skin tone.  Used by FaceShifter, DeepFaceLab and Adobe's Content-Aware Fill.

    strength: 0 = no change, 1 = full match to target colour
    """
    # Erode mask to get the 'inner face' region; subtract = outer ring
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    inner  = cv2.erode(mask, k, iterations=2)
    ring   = cv2.subtract(mask, inner)                      # boundary zone only
    ring_f = ring.astype(np.float32) / 255.0

    sw_lab  = cv2.cvtColor(swapped,  cv2.COLOR_BGR2LAB).astype(np.float64)
    tg_lab  = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float64)

    face_px = mask > 30
    if not face_px.any():
        return swapped

    result_lab = sw_lab.copy()
    for c in range(3):
        sw_vals = sw_lab[:, :, c][face_px]
        tg_vals = tg_lab[:, :, c][face_px]
        sw_mu, sw_sig = sw_vals.mean(), sw_vals.std() + 1e-6
        tg_mu, tg_sig = tg_vals.mean(), tg_vals.std() + 1e-6
        # Normalise source statistics to target statistics
        corrected = (sw_lab[:, :, c] - sw_mu) * (tg_sig / sw_sig) + tg_mu
        # Blend: full correction at ring, zero at inner face
        result_lab[:, :, c] = (corrected * ring_f * strength
                                + sw_lab[:, :, c] * (1.0 - ring_f * strength))

    corrected_bgr = cv2.cvtColor(
        np.clip(result_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Apply only inside mask to avoid touching the background
    m3 = np.dstack([ring_f] * 3)
    return (corrected_bgr.astype(np.float32) * m3
            + swapped.astype(np.float32) * (1.0 - m3)).astype(np.uint8)


def _laplacian_pyramid_blend(img_a: np.ndarray, img_b: np.ndarray,
                               mask: np.ndarray, levels: int = 6) -> np.ndarray:
    """
    Multi-scale Laplacian pyramid blending (Burt & Adelson 1983).
    Used in: Adobe Photoshop, DeepFaceLab, computational photography pipelines.

    Blends img_a (swapped) into img_b (original) at every spatial scale,
    so there are no seam artefacts at any frequency — unlike alpha-blend
    (sharp at fine scales) or Poisson clone (breaks on large colour gaps).
    """
    m  = mask.astype(np.float32) / 255.0
    m3 = np.dstack([m, m, m])
    A  = img_a.astype(np.float32)
    B  = img_b.astype(np.float32)

    # ── Gaussian pyramids ────────────────────────────────────────────────────
    gp_a, gp_b, gp_m = [A], [B], [m3]
    for _ in range(levels):
        gp_a.append(cv2.pyrDown(gp_a[-1]))
        gp_b.append(cv2.pyrDown(gp_b[-1]))
        gp_m.append(cv2.pyrDown(gp_m[-1]))

    # ── Laplacian pyramids ───────────────────────────────────────────────────
    lp_a = [gp_a[levels]]
    lp_b = [gp_b[levels]]
    for i in range(levels, 0, -1):
        h_, w_ = gp_a[i - 1].shape[:2]
        lp_a.append(gp_a[i - 1] - cv2.pyrUp(gp_a[i], dstsize=(w_, h_)))
        lp_b.append(gp_b[i - 1] - cv2.pyrUp(gp_b[i], dstsize=(w_, h_)))

    # ── Blend each pyramid level ─────────────────────────────────────────────
    blended = []
    for la, lb, gm in zip(lp_a, lp_b, reversed(gp_m)):
        if gm.shape[:2] != la.shape[:2]:
            gm = cv2.resize(gm, (la.shape[1], la.shape[0]))
        blended.append(la * gm + lb * (1.0 - gm))

    # ── Collapse ─────────────────────────────────────────────────────────────
    out = blended[0]
    for bp in blended[1:]:
        h_, w_ = bp.shape[:2]
        out = cv2.pyrUp(out, dstsize=(w_, h_)) + bp

    return np.clip(out, 0, 255).astype(np.uint8)


def _advanced_face_blend(swapped: np.ndarray, original: np.ndarray, faces) -> np.ndarray:
    """
    Two-stage professional blending pipeline applied after inswapper + GFPGAN:

    Stage 1 — LAB boundary colour transfer  (closes the skin-tone gap)
    Stage 2 — Laplacian pyramid compositing (seam-free at all spatial scales)

    Falls back to soft alpha-blend if anything raises.
    """
    if not faces:
        return swapped

    h, w   = original.shape[:2]
    result = swapped.copy()

    for face in faces:
        try:
            mask = _build_ellipse_mask(h, w, face)

            # Stage 1: pull the boundary pixels toward the target skin tone
            result = _lab_colour_transfer(result, original, mask, strength=0.55)

            # Stage 2: multi-scale pyramid blend for artefact-free seam
            result = _laplacian_pyramid_blend(result, original, mask, levels=6)

        except Exception as exc:
            logger.warning(f'Advanced blend failed ({exc}) — alpha fallback')
            try:
                x1, y1, x2, y2 = [int(v) for v in face.bounding_box]
                a = np.zeros((h, w, 1), dtype=np.float32)
                a[y1:y2, x1:x2] = 1.0
                a = cv2.GaussianBlur(a, (61, 61), 0)
                result = (swapped.astype(np.float32) * a
                          + original.astype(np.float32) * (1.0 - a)).astype(np.uint8)
            except Exception:
                result = swapped

    return result


# ── Face consistency tracking ─────────────────────────────────────────────────
def _bbox_iou(a, b) -> float:
    """Intersection-over-Union for two [x1,y1,x2,y2] bounding boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _match_faces(new_faces, prev_faces):
    """
    Reorder new_faces to best match prev_faces by IoU overlap.
    Keeps face-slot assignment stable across frames.
    """
    if not prev_faces or not new_faces:
        return new_faces
    matched, used = [], set()
    for pf in prev_faces:
        best_iou, best_idx = 0.0, -1
        for i, nf in enumerate(new_faces):
            if i in used:
                continue
            iou = _bbox_iou(pf.bounding_box, nf.bounding_box)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_idx >= 0 and best_iou > 0.15:
            matched.append(new_faces[best_idx])
            used.add(best_idx)
    # Append any new faces not matched to previous slots
    for i, nf in enumerate(new_faces):
        if i not in used:
            matched.append(nf)
    return matched if matched else new_faces


# ── Core face-swap logic ──────────────────────────────────────────────────────
def read_frame(path: str) -> np.ndarray | None:
    """Read an image/first-video-frame from disk."""
    if is_image(path):
        return read_static_image(path)
    if is_video(path):
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    return None


def detect_faces_in_frame(frame: np.ndarray):
    """Return list of detected Face objects sorted left→right."""
    if frame is None or not np.any(frame):
        return []
    faces = face_analyser.get_many_faces([frame])
    # Sort left-to-right for consistent indexing shown in UI
    faces.sort(key=lambda f: f.bounding_box[0])
    return faces


def swap_frame(source_face, target_faces, frame: np.ndarray,
               face_indices=None, enhance: bool = True) -> np.ndarray:
    """
    Apply `source_face` onto selected `target_faces` in `frame`.
    `face_indices`: list of int indices into `target_faces` to swap (None = all).
    """
    result = frame.copy()
    for idx, tface in enumerate(target_faces):
        if face_indices is not None and idx not in face_indices:
            continue
        try:
            result = face_swapper.swap_face(source_face, tface, result)
            if enhance:
                result = face_enhancer.enhance_face(tface, result)
        except Exception as e:
            logger.warning(f'swap/enhance failed for face {idx}: {e}')
    return result


def _resize_frame(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max_side <= 0 or max(w, h) <= max_side:
        return frame
    scale = max_side / max(w, h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)


def process_image_swap(source_path: str, target_path: str, output_path: str,
                       face_indices=None, enhance: bool = True,
                       pitch_semitones: float = 0.0,
                       job_id: str = '',
                       detect_interval: int = 5,
                       max_side: int = 720) -> dict:
    """
    Full pipeline for image→image or video→video face swap.
    detect_interval: re-detect faces every N frames (1 = every frame).
    max_side: downscale video so longest side ≤ this (0 = no downscale).
    """
    t0 = time.time()

    # ── Read source ──────────────────────────────────────────────────────────
    source_frame = read_frame(source_path)
    if source_frame is None:
        return {'success': False, 'error': 'Cannot read source file'}

    state_manager.set_item('source_paths', [source_path])

    source_faces = detect_faces_in_frame(source_frame)
    if not source_faces:
        return {
            'success': False,
            'error': (
                'No face detected in the source image. '
                'Tips: use a clear front-facing portrait, '
                'ensure good lighting, and avoid heavy occlusion.'
            )
        }
    source_face = source_faces[0]
    logger.info(f'Source face detected – score {source_face.score_set.get("detector"):.3f}')

    # ── Image swap ───────────────────────────────────────────────────────────
    if is_image(target_path):
        # Use highest pixel boost for images — quality over speed
        state_manager.set_item('face_swapper_pixel_boost', '256x256')
        target_frame = read_frame(target_path)
        if target_frame is None:
            return {'success': False, 'error': 'Cannot read target image'}

        target_faces = detect_faces_in_frame(target_frame)
        if not target_faces:
            return {'success': False, 'error': 'No face detected in the target image'}

        logger.info(f'{len(target_faces)} target face(s) detected')
        result = swap_frame(source_face, target_faces, target_frame,
                            face_indices=face_indices, enhance=enhance)

        # Stage 3: LAB colour transfer + Laplacian pyramid blend
        active_faces = [target_faces[i] for i in range(len(target_faces))
                        if face_indices is None or i in face_indices]
        result = _advanced_face_blend(result, target_frame, active_faces)

        ok = write_image(output_path, result)
        if not ok:
            return {'success': False, 'error': 'Failed to write output image'}

        return {
            'success': True,
            'processing_time': round(time.time() - t0, 2),
            'faces_swapped': len([i for i in range(len(target_faces))
                                  if face_indices is None or i in face_indices]),
            'output_type': 'image'
        }

    # ── Video swap ───────────────────────────────────────────────────────────
    if is_video(target_path):
        cap = cv2.VideoCapture(target_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Compute output dimensions with optional downscale
        if max_side > 0 and max(src_w, src_h) > max_side:
            scale = max_side / max(src_w, src_h)
            out_w, out_h = int(src_w * scale) & ~1, int(src_h * scale) & ~1  # even dims for H.264
        else:
            out_w, out_h = src_w & ~1, src_h & ~1
        logger.info(f'Video: {src_w}×{src_h} → {out_w}×{out_h}, {total} frames @ {fps:.1f}fps, detect every {detect_interval} frames')
        # Use 128x128 for video — speed matters more than pixel-perfect quality per frame
        state_manager.set_item('face_swapper_pixel_boost', '128x128')

        base, _ = os.path.splitext(output_path)
        tmp_video = base + '_raw.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(tmp_video, fourcc, fps, (out_w, out_h))

        # Validate first frame has a face
        ok_first, first_frame = cap.read()
        if not ok_first:
            cap.release()
            return {'success': False, 'error': 'Cannot read first video frame'}
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        first_resized = _resize_frame(first_frame, max_side)
        target_faces_first = detect_faces_in_frame(first_resized)
        if not target_faces_first:
            cap.release()
            return {'success': False, 'error': 'No face detected in target video'}

        if job_id:
            _job_progress[job_id] = {'total': total or 1, 'done': 0, 'status': 'processing',
                                      'eta_seconds': 0, 'fps_proc': 0.0}

        cached_faces = target_faces_first
        frame_count  = 0
        skip_count   = 0
        t_loop_start = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = _resize_frame(frame, max_side)

            # Re-detect faces every N frames; reuse cache otherwise
            if frame_count % detect_interval == 0:
                try:
                    detected = detect_faces_in_frame(frame)
                    if detected:
                        # Reorder to match previous face slots (avoids identity swap jitter)
                        cached_faces = _match_faces(detected, cached_faces)
                except Exception as det_err:
                    logger.warning(f'Frame {frame_count}: detection failed – {det_err}')

            if cached_faces:
                try:
                    frame = swap_frame(source_face, cached_faces, frame,
                                       face_indices=face_indices, enhance=enhance)
                except Exception as swap_err:
                    # Write original frame on failure rather than aborting
                    skip_count += 1
                    logger.warning(f'Frame {frame_count}: swap failed (skipped) – {swap_err}')

            writer.write(frame)
            frame_count += 1

            # Update progress + live ETA every 3 frames
            if job_id and frame_count % 3 == 0:
                elapsed  = time.time() - t_loop_start
                fps_proc = frame_count / elapsed if elapsed > 0 else 0
                remaining = (total - frame_count) / fps_proc if fps_proc > 0 and total > frame_count else 0
                _job_progress[job_id] = {
                    'total': total or frame_count,
                    'done': frame_count,
                    'status': 'processing',
                    'eta_seconds': int(remaining),
                    'fps_proc': round(fps_proc, 2),
                    'skipped': skip_count,
                }

        cap.release()
        writer.release()

        if job_id:
            _job_progress[job_id] = {'total': frame_count, 'done': frame_count,
                                      'status': 'encoding', 'eta_seconds': 0, 'fps_proc': 0}

        # Re-encode to H.264 + mux audio
        if _ffmpeg_available():
            ok_mux = _mux_video_audio(tmp_video, target_path, output_path, pitch_semitones)
            if not ok_mux:
                logger.warning('ffmpeg mux failed – falling back to video-only copy')
                import shutil; shutil.move(tmp_video, output_path)
        else:
            import shutil; shutil.move(tmp_video, output_path)

        if Path(tmp_video).exists() and tmp_video != output_path:
            try: Path(tmp_video).unlink()
            except Exception: pass

        if job_id:
            _job_progress[job_id] = {'total': frame_count, 'done': frame_count,
                                      'status': 'done', 'eta_seconds': 0, 'fps_proc': 0}

        elapsed_total = time.time() - t0
        return {
            'success': True,
            'processing_time': round(elapsed_total, 2),
            'frames_processed': frame_count,
            'frames_skipped': skip_count,
            'fps_achieved': round(frame_count / elapsed_total, 2) if elapsed_total > 0 else 0,
            'faces_swapped': len([i for i in range(len(target_faces_first))
                                  if face_indices is None or i in face_indices]),
            'output_type': 'video',
            'output_resolution': f'{out_w}x{out_h}',
        }

    return {'success': False, 'error': 'Unsupported target file type'}


# ── Webcam state (thread-safe) ────────────────────────────────────────────────
_webcam_lock = threading.Lock()
_webcam_state = {
    'source_face': None,     # current source Face object
    'source_path': None,     # path of uploaded source image
    'last_result': None,     # most recent swapped frame (BGR ndarray)
    'frame_count': 0,
}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/detect-faces', methods=['POST'])
def api_detect_faces():
    """
    Detect faces in an uploaded image.
    Returns list of face crops (base64 JPEG thumbnails) so the user can
    select which faces to swap.
    """
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'}), 400

    img_file = request.files['image']
    if not allowed_file(img_file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400

    fname = secure_filename(f'detect_{int(time.time())}_{img_file.filename}')
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    img_file.save(path)

    frame = read_frame(path)
    if frame is None:
        return jsonify({'success': False, 'error': 'Cannot read image'}), 400

    faces = detect_faces_in_frame(frame)
    crops = []
    for face in faces:
        crop = _crop_face(frame, face)
        crops.append({
            'b64': _frame_to_b64(crop),
            'score': float(face.score_set.get('detector', 0)),
            'bbox': [float(v) for v in face.bounding_box],
        })

    return jsonify({
        'success': True,
        'count': len(faces),
        'faces': crops,
        'image_path': fname,
    })


@app.route('/api/swap', methods=['POST'])
def api_swap():
    """
    Perform face swap.
    Form fields:
      source          – source image file
      target          – target image or video file
      face_indices    – JSON array of int indices (omit → swap all faces)
      enhance         – '1' to enable face enhancer (default '1')
      pitch_semitones – float, audio pitch shift in semitones (video only)
    """
    if 'source' not in request.files or 'target' not in request.files:
        return jsonify({'success': False, 'error': 'Missing source or target file'}), 400

    src_file = request.files['source']
    tgt_file = request.files['target']

    if not (allowed_file(src_file.filename) and allowed_file(tgt_file.filename)):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400

    ts = int(time.time())
    src_name = secure_filename(f'source_{ts}_{src_file.filename}')
    tgt_name = secure_filename(f'target_{ts}_{tgt_file.filename}')
    src_path = os.path.join(app.config['UPLOAD_FOLDER'], src_name)
    tgt_path = os.path.join(app.config['UPLOAD_FOLDER'], tgt_name)
    src_file.save(src_path)
    tgt_file.save(tgt_path)

    # Parse options
    face_indices_raw = request.form.get('face_indices', '')
    face_indices = None
    if face_indices_raw.strip():
        try:
            import json
            face_indices = list(set(int(i) for i in json.loads(face_indices_raw)))
        except Exception:
            pass

    enhance = request.form.get('enhance', '1') == '1'

    try:
        pitch = float(request.form.get('pitch_semitones', '0'))
    except ValueError:
        pitch = 0.0

    job_id = request.form.get('job_id', '')
    if job_id:
        _job_progress[job_id] = {'total': 0, 'done': 0, 'status': 'uploaded',
                                  'eta_seconds': 0, 'fps_proc': 0.0}

    try:
        detect_interval = max(1, int(request.form.get('detect_interval', '5')))
    except ValueError:
        detect_interval = 5
    try:
        max_side = int(request.form.get('max_side', '720'))
    except ValueError:
        max_side = 720

    # Preserve original image/video format
    tgt_ext = os.path.splitext(tgt_file.filename)[1].lower()
    valid_image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    valid_video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    if tgt_ext in valid_image_exts:
        ext = tgt_ext
    elif tgt_ext in valid_video_exts:
        ext = '.mp4'
    else:
        ext = '.jpg'
    out_name = f'output_{ts}{ext}'
    out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_name)

    result = process_image_swap(src_path, tgt_path, out_path,
                                face_indices=face_indices,
                                enhance=enhance,
                                pitch_semitones=pitch,
                                job_id=job_id,
                                detect_interval=detect_interval,
                                max_side=max_side)
    if result['success']:
        result['output_filename'] = out_name
        result['output_url'] = f'/output/{out_name}'
        # Embed image as base64 so the browser can display it without a
        # separate HTTP request (avoids any file-serving path issues)
        if result.get('output_type') == 'image' and os.path.isfile(out_path):
            with open(out_path, 'rb') as f:
                result['preview_b64'] = base64.b64encode(f.read()).decode()
    return jsonify(result), 200 if result['success'] else 500


@app.route('/api/webcam/set-source', methods=['POST'])
def webcam_set_source():
    """Upload the source face to use for webcam swapping."""
    if 'source' not in request.files:
        return jsonify({'success': False, 'error': 'No source file'}), 400

    src_file = request.files['source']
    if not allowed_file(src_file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400

    ts = int(time.time())
    fname = secure_filename(f'webcam_src_{ts}_{src_file.filename}')
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    src_file.save(path)

    frame = read_frame(path)
    if frame is None:
        return jsonify({'success': False, 'error': 'Cannot read source image'}), 400

    faces = detect_faces_in_frame(frame)
    if not faces:
        return jsonify({
            'success': False,
            'error': 'No face detected in source image. Use a clear front-facing portrait.'
        }), 400

    state_manager.set_item('source_paths', [path])
    with _webcam_lock:
        _webcam_state['source_face'] = faces[0]
        _webcam_state['source_path'] = path
        _webcam_state['last_result'] = None

    crop = _crop_face(frame, faces[0])
    return jsonify({
        'success': True,
        'preview_b64': _frame_to_b64(crop),
        'score': float(faces[0].score_set.get('detector', 0)),
    })


@app.route('/api/webcam/process-frame', methods=['POST'])
def webcam_process_frame():
    """
    Process a single webcam frame.
    JSON body: { "frame_b64": "<JPEG base64>", "enhance": false }
    Returns: { "result_b64": "<JPEG base64>", "fps": 3.2 }
    """
    data = request.get_json(force=True, silent=True) or {}
    frame_b64 = data.get('frame_b64', '')
    enhance = bool(data.get('enhance', False))  # disable by default for speed

    if not frame_b64:
        return jsonify({'success': False, 'error': 'No frame data'}), 400

    with _webcam_lock:
        source_face = _webcam_state.get('source_face')

    if source_face is None:
        return jsonify({'success': False, 'error': 'No source face set – upload one first'}), 400

    t0 = time.perf_counter()
    frame = _b64_to_frame(frame_b64)
    if frame is None:
        return jsonify({'success': False, 'error': 'Cannot decode frame'}), 400

    tfaces = detect_faces_in_frame(frame)
    if tfaces:
        frame = swap_frame(source_face, tfaces, frame, enhance=enhance)

    elapsed = time.perf_counter() - t0
    fps = round(1.0 / max(elapsed, 0.001), 1)

    result_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return jsonify({
        'success': True,
        'result_b64': _frame_to_b64(result_rgb, quality=70),
        'fps': fps,
        'faces_found': len(tfaces),
    })


@app.route('/output/<path:filename>')
def get_output(filename: str):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


@app.route('/api/progress/<job_id>')
def api_progress(job_id: str):
    p = _job_progress.get(job_id)
    if p is None:
        return jsonify({'found': False})
    return jsonify({'found': True, **p})


@app.route('/api/active-jobs')
def api_active_jobs():
    """Return any jobs that are currently processing (for reconnect after refresh)."""
    active = [
        {'job_id': jid, **info}
        for jid, info in _job_progress.items()
        if info.get('status') in ('processing', 'encoding', 'uploaded')
    ]
    return jsonify({'jobs': active})


# ── Backward-compatible /swap route ───────────────────────────────────────────
@app.route('/swap', methods=['POST'])
def swap_faces_compat():
    """Legacy endpoint kept for backward compatibility."""
    return api_swap()


# ── HTML template (single-page, tabbed) ──────────────────────────────────────
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepTrace v2 – Face Swap Studio</title>
<style>
  :root {
    --accent: #7c5cfc;
    --accent2: #5b3de8;
    --bg: #0f0f1a;
    --surface: #1a1a2e;
    --surface2: #16213e;
    --text: #e2e2f0;
    --muted: #7f7fa0;
    --success: #22c55e;
    --error: #ef4444;
    --warn: #f59e0b;
    --radius: 12px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }

  .header { background: var(--surface); padding: 18px 24px; border-bottom: 1px solid #ffffff15;
            display: flex; align-items: center; gap: 14px; }
  .header h1 { font-size: 1.5rem; background: linear-gradient(135deg,#a78bfa,#60a5fa); -webkit-background-clip:text; color:transparent; }
  .badge { background: var(--accent); color: #fff; font-size: .65rem; padding: 2px 8px; border-radius: 999px; font-weight: 700; }

  .tabs { display: flex; gap: 4px; padding: 16px 24px 0; background: var(--surface); }
  .tab { padding: 10px 22px; border-radius: var(--radius) var(--radius) 0 0; cursor: pointer;
         background: var(--bg); color: var(--muted); font-weight: 600; transition: .2s; border: none; font-size: .9rem; }
  .tab.active { background: var(--accent); color: #fff; }

  .panel { display: none; padding: 28px 24px; max-width: 1080px; margin: 0 auto; }
  .panel.active { display: block; }

  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media(max-width:640px){ .grid2 { grid-template-columns:1fr; } }

  .card { background: var(--surface); border-radius: var(--radius); padding: 20px; }
  .card h3 { font-size: .85rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 14px; }

  label { display: block; margin-bottom: 6px; font-size: .9rem; color: var(--muted); }
  .upload-zone { border: 2px dashed #ffffff25; border-radius: 10px; padding: 20px;
                 text-align: center; cursor: pointer; transition: .2s; background: var(--surface2); }
  .upload-zone:hover, .upload-zone.drag { border-color: var(--accent); background: #7c5cfc15; }
  .upload-zone input { display:none; }
  .upload-zone .icon { font-size: 2rem; margin-bottom: 8px; }
  .upload-zone p { font-size: .85rem; color: var(--muted); }
  .preview-img { max-width: 100%; max-height: 220px; border-radius: 8px; margin-top: 10px; display:none; }

  .face-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
  .face-thumb { width: 80px; height: 80px; border-radius: 8px; object-fit: cover; cursor: pointer;
                border: 3px solid transparent; transition: .15s; }
  .face-thumb.selected { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent); }

  .btn { display: inline-block; padding: 12px 28px; border-radius: 8px; border: none; cursor: pointer;
         font-weight: 700; font-size: .95rem; transition: .2s; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:hover { background: var(--accent2); transform: translateY(-1px); }
  .btn-primary:disabled { opacity: .55; cursor: not-allowed; transform: none; }
  .btn-sm { padding: 7px 16px; font-size: .8rem; border-radius: 6px; }
  .btn-success { background: var(--success); color: #fff; }
  .btn-outline { background: transparent; color: var(--text); border: 1px solid #ffffff30; }
  .btn-outline:hover { border-color: var(--accent); color: var(--accent); }

  .alert { padding: 12px 16px; border-radius: 8px; margin-top: 16px; font-size: .9rem; display:none; }
  .alert.error   { background:#ef444420; border:1px solid #ef444460; color:#fca5a5; }
  .alert.success { background:#22c55e20; border:1px solid #22c55e60; color:#86efac; }
  .alert.info    { background:#60a5fa20; border:1px solid #60a5fa60; color:#93c5fd; }
  .alert.warn    { background:#f59e0b20; border:1px solid #f59e0b60; color:#fcd34d; }

  .output-box { margin-top: 20px; background: var(--surface2); border-radius: 10px; padding: 16px; display:none; }
  .output-box img, .output-box video { max-width:100%; border-radius:8px; }
  .output-meta { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 10px; font-size: .8rem; color: var(--muted); }
  .stat { background: var(--surface); padding: 4px 10px; border-radius: 6px; }

  input[type=range] { width:100%; accent-color: var(--accent); }
  .range-row { display:flex; align-items:center; gap:10px; }
  .range-val { min-width: 2.5em; text-align:center; font-weight:700; color:var(--accent); }

  /* Webcam tab */
  .webcam-layout { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media(max-width:700px){ .webcam-layout { grid-template-columns:1fr; } }
  #webcamVideo { width:100%; border-radius: 10px; background:#000; max-height:360px; }
  #webcamResult { width:100%; border-radius: 10px; background:#111; max-height:360px; display:block; }
  .cam-controls { display:flex; gap:10px; flex-wrap:wrap; margin-top:10px; align-items:center; }
  .cam-badge { background: var(--surface); border-radius:6px; padding:4px 10px; font-size:.8rem; color:var(--muted); }
  #srcPreviewThumb { width:64px; height:64px; border-radius:8px; object-fit:cover; display:none; }

  .spinner { display:inline-block; width:18px; height:18px; border:3px solid #ffffff40;
             border-top-color: var(--accent); border-radius:50%; animation:spin .7s linear infinite; vertical-align:middle; }
  @keyframes spin { to { transform:rotate(360deg); } }

  .progress { background:#ffffff15; border-radius:999px; height:6px; overflow:hidden; margin-top:12px; display:none; }
  .progress-bar { height:100%; background: linear-gradient(90deg,var(--accent),#60a5fa); border-radius:999px;
                  transition: width .3s; }

  footer { text-align:center; padding:24px; color:var(--muted); font-size:.8rem; }
</style>
</head>
<body>

<div class="header">
  <span style="font-size:1.8rem">🎭</span>
  <h1>DeepTrace</h1>
  <span class="badge">v2.0</span>
  <span style="margin-left:auto; color:var(--muted); font-size:.85rem">Face Swap Studio</span>
</div>

<div class="tabs">
  <button class="tab active" onclick="switchTab('swap')">🔄 Image / Video Swap</button>
  <button class="tab" onclick="switchTab('webcam')">📷 Live Webcam</button>
  <button class="tab" onclick="switchTab('multi')">👥 Multi-Face Selector</button>
</div>

<!-- ══════════ TAB: Image/Video Swap ══════════ -->
<div id="tab-swap" class="panel active">
  <div class="grid2">
    <div class="card">
      <h3>Source Face</h3>
      <div class="upload-zone" id="srcZone" onclick="document.getElementById('srcFile').click()">
        <div class="icon">🤳</div>
        <p>Click or drag a portrait photo</p>
        <input type="file" id="srcFile" accept="image/*">
      </div>
      <img id="srcPreview" class="preview-img">
    </div>

    <div class="card">
      <h3>Target (Image or Video)</h3>
      <div class="upload-zone" id="tgtZone" onclick="document.getElementById('tgtFile').click()">
        <div class="icon">🖼️</div>
        <p>Click or drag an image or video</p>
        <input type="file" id="tgtFile" accept="image/*,video/*">
      </div>
      <img id="tgtPreview" class="preview-img">
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h3>Options</h3>
    <div class="grid2">
      <div>
        <label>Face Enhancer</label>
        <label style="display:flex;align-items:center;gap:8px;margin-top:4px">
          <input type="checkbox" id="enhanceCheck" checked style="accent-color:var(--accent)">
          <span style="font-size:.9rem">Enable GFPGAN (slower, higher quality)</span>
        </label>
        <div style="font-size:.75rem;color:var(--muted);margin-top:4px" id="enhanceHint">ON by default for images — auto-disabled for video</div>
      </div>
      <div>
        <label>Audio Pitch Shift (video only)</label>
        <div class="range-row">
          <input type="range" id="pitchRange" min="-12" max="12" step="1" value="0"
                 oninput="document.getElementById('pitchVal').textContent=this.value">
          <span class="range-val" id="pitchVal">0</span>
          <span style="font-size:.8rem;color:var(--muted)">semitones</span>
        </div>
      </div>
    </div>

    <div id="videoOpts" style="display:none;margin-top:16px;padding-top:14px;border-top:1px solid #ffffff12">
      <div style="font-size:.8rem;color:var(--accent);font-weight:700;margin-bottom:10px">⚡ Video Speed Settings</div>
      <div class="grid2">
        <div>
          <label>Face Detect Interval</label>
          <select id="detectInterval" style="width:100%;background:var(--surface2);color:var(--text);border:1px solid #ffffff20;border-radius:6px;padding:8px;font-size:.9rem">
            <option value="1">Every frame (slowest, most accurate)</option>
            <option value="3" selected>Every 3 frames (recommended)</option>
            <option value="5">Every 5 frames (faster)</option>
            <option value="10">Every 10 frames (fastest)</option>
            <option value="30">Every 30 frames (ultra fast)</option>
          </select>
          <div style="font-size:.73rem;color:var(--muted);margin-top:4px">Skip face re-detection between frames</div>
        </div>
        <div>
          <label>Max Resolution</label>
          <select id="maxSide" style="width:100%;background:var(--surface2);color:var(--text);border:1px solid #ffffff20;border-radius:6px;padding:8px;font-size:.9rem">
            <option value="480">480p (fastest)</option>
            <option value="720" selected>720p (recommended)</option>
            <option value="1080">1080p</option>
            <option value="0">Original (no resize)</option>
          </select>
          <div style="font-size:.73rem;color:var(--muted);margin-top:4px">Downscale before processing</div>
        </div>
      </div>
      <div id="timeEstimate" style="margin-top:12px;padding:10px 14px;background:var(--surface2);border-radius:8px;font-size:.85rem;color:var(--muted)">
        Select a video to see estimated processing time
      </div>
    </div>
  </div>

  <div style="margin-top:20px;display:flex;gap:12px;flex-wrap:wrap">
    <button class="btn btn-primary" id="swapBtn" onclick="doSwap()">⚡ Swap Faces</button>
    <button class="btn btn-outline" onclick="resetSwap()">✖ Reset</button>
  </div>

  <div id="swapAlert" class="alert"></div>
  <div class="progress" id="swapProgress"><div class="progress-bar" id="swapBar" style="width:0%"></div></div>

  <div class="output-box" id="swapOutput">
    <h3 style="font-size:.85rem;color:var(--muted);margin-bottom:10px">RESULT</h3>
    <img id="outImg" style="display:none;max-width:100%;border-radius:10px;box-shadow:0 4px 20px rgba(0,0,0,.4)">
    <video id="outVid" controls style="display:none;max-width:100%;border-radius:10px"></video>
    <div class="output-meta" id="outMeta"></div>
    <div style="margin-top:12px; display:flex; gap:10px; align-items:center; flex-wrap:wrap">
      <a id="outDl" class="btn btn-success btn-sm" download>⬇ Download Media</a>
      <a id="outDlHtml" class="btn btn-secondary btn-sm" style="display:none" download>📄 Download as HTML</a>
    </div>
  </div>
</div>

<!-- ══════════ TAB: Live Webcam ══════════ -->
<div id="tab-webcam" class="panel">
  <div class="grid2" style="margin-bottom:16px">
    <div class="card">
      <h3>Source Face for Webcam</h3>
      <div class="upload-zone" id="wcSrcZone" onclick="document.getElementById('wcSrcFile').click()">
        <div class="icon">🤳</div>
        <p>Upload a portrait for the swap</p>
        <input type="file" id="wcSrcFile" accept="image/*">
      </div>
      <div style="margin-top:10px;display:flex;align-items:center;gap:10px">
        <img id="srcPreviewThumb">
        <div id="wcSrcStatus" style="font-size:.85rem;color:var(--muted)">No source set</div>
      </div>
    </div>

    <div class="card">
      <h3>Enhancement Options</h3>
      <label style="display:flex;align-items:center;gap:8px;margin-top:4px">
        <input type="checkbox" id="wcEnhance" style="accent-color:var(--accent)">
        <span style="font-size:.9rem">Enable enhancer (slower)</span>
      </label>
      <div style="margin-top:14px;font-size:.85rem;color:var(--muted)">
        💡 Webcam swap runs at ~2–5 fps on CPU. Disable enhancer for better real-time speed.
      </div>
      <div style="margin-top:10px">
        <button class="btn btn-outline btn-sm" onclick="toggleCamera()" id="camToggleBtn">▶ Start Camera</button>
        <button class="btn btn-outline btn-sm" onclick="stopWebcam()" id="camStopBtn" style="display:none">⏹ Stop</button>
      </div>
    </div>
  </div>

  <div class="webcam-layout">
    <div>
      <h3 style="font-size:.8rem;color:var(--muted);margin-bottom:8px">📹 ORIGINAL</h3>
      <video id="webcamVideo" autoplay muted playsinline></video>
      <canvas id="captureCanvas" style="display:none"></canvas>
    </div>
    <div>
      <h3 style="font-size:.8rem;color:var(--muted);margin-bottom:8px">🎭 SWAPPED</h3>
      <img id="webcamResult" src="" alt="waiting…">
      <div class="cam-controls">
        <span class="cam-badge" id="fpsBadge">– fps</span>
        <span class="cam-badge" id="facesBadge">– faces</span>
      </div>
    </div>
  </div>
  <div id="wcAlert" class="alert"></div>
</div>

<!-- ══════════ TAB: Multi-Face Selector ══════════ -->
<div id="tab-multi" class="panel">
  <div class="grid2">
    <div class="card">
      <h3>Source Face</h3>
      <div class="upload-zone" onclick="document.getElementById('mfSrcFile').click()">
        <div class="icon">🤳</div><p>Portrait photo</p>
        <input type="file" id="mfSrcFile" accept="image/*">
      </div>
      <img id="mfSrcPreview" class="preview-img">
    </div>

    <div class="card">
      <h3>Target Image (multi-face)</h3>
      <div class="upload-zone" onclick="document.getElementById('mfTgtFile').click()">
        <div class="icon">👥</div><p>Group photo or image with multiple faces</p>
        <input type="file" id="mfTgtFile" accept="image/*">
      </div>
      <img id="mfTgtPreview" class="preview-img">
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h3>Detected Target Faces – click to toggle selection</h3>
    <div id="mfFaceGrid" class="face-grid">
      <span style="color:var(--muted);font-size:.85rem">Upload a target image to detect faces…</span>
    </div>
    <div style="margin-top:12px;font-size:.82rem;color:var(--muted)" id="mfSelInfo">
      0 faces selected → all will be swapped
    </div>
  </div>

  <div style="margin-top:20px;display:flex;gap:12px;flex-wrap:wrap">
    <button class="btn btn-primary" onclick="doMultiSwap()" id="mfSwapBtn">⚡ Swap Selected Faces</button>
    <button class="btn btn-outline btn-sm" onclick="selectAllFaces()">Select All</button>
    <button class="btn btn-outline btn-sm" onclick="deselectAllFaces()">Deselect All</button>
  </div>

  <div id="mfAlert" class="alert"></div>
  <div class="output-box" id="mfOutput">
    <h3 style="font-size:.85rem;color:var(--muted);margin-bottom:10px">RESULT</h3>
    <img id="mfOutImg" style="max-width:100%;border-radius:10px;box-shadow:0 4px 20px rgba(0,0,0,.4)">
    <div class="output-meta" id="mfOutMeta"></div>
    <div style="margin-top:12px; display:flex; gap:10px; align-items:center; flex-wrap:wrap">
      <a id="mfOutDl" class="btn btn-success btn-sm" download>⬇ Download Media</a>
      <a id="mfOutDlHtml" class="btn btn-secondary btn-sm" style="display:none" download>📄 Download as HTML</a>
    </div>
  </div>
</div>

<!-- Reconnect banner (hidden by default) -->
<div id="reconnectBanner" style="display:none;position:fixed;bottom:0;left:0;right:0;z-index:999;
  background:linear-gradient(90deg,#1a1a2e,#16213e);border-top:1px solid var(--accent);
  padding:14px 24px;display:none;align-items:center;gap:16px;flex-wrap:wrap">
  <span style="font-size:1.1rem">⚠️</span>
  <div style="flex:1">
    <div style="font-weight:700;color:var(--accent)" id="reconnectTitle">Job running on server</div>
    <div style="font-size:.82rem;color:var(--muted)" id="reconnectInfo">Your video is still being processed</div>
  </div>
  <button class="btn btn-primary btn-sm" onclick="reconnectJob()">▶ Reconnect</button>
  <button class="btn btn-outline btn-sm" onclick="dismissReconnect()">✕ Dismiss</button>
</div>

<footer>DeepTrace v2 &nbsp;|&nbsp; FYP Research Project &nbsp;|&nbsp; Powered by InsightFace + ONNX Runtime</footer>

<script>
// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  const tabs = ['swap','webcam','multi'];
  const idx = tabs.indexOf(name);
  document.querySelectorAll('.tab')[idx].classList.add('active');
  document.getElementById('tab-'+name).classList.add('active');
  if(name!=='webcam') stopWebcam();
}

// ── Job reconnect (survives page refresh) ─────────────────────────────────────
let _activeJobId = null;
let _reconnectPoll = null;

async function checkForActiveJobs(){
  try{
    const res = await fetch('/api/active-jobs');
    const data = await res.json();
    if(data.jobs && data.jobs.length > 0){
      const job = data.jobs[0];
      const saved = localStorage.getItem('deeptrace_job_id');
      // Prefer saved job if it matches; otherwise take first active
      const matchedJob = data.jobs.find(j=>j.job_id===saved) || job;
      showReconnectBanner(matchedJob);
    } else {
      localStorage.removeItem('deeptrace_job_id');
    }
  } catch(e){}
}

function showReconnectBanner(job){
  _activeJobId = job.job_id;
  const banner = document.getElementById('reconnectBanner');
  banner.style.display = 'flex';
  const pct = job.total > 0 ? Math.round(job.done/job.total*100) : 0;
  const eta = job.eta_seconds > 0
    ? (job.eta_seconds < 60 ? `${job.eta_seconds}s left` : `${Math.round(job.eta_seconds/60)}m left`)
    : '';
  document.getElementById('reconnectInfo').textContent =
    `Frame ${job.done}/${job.total} (${pct}%) ${eta} — click Reconnect to resume progress view`;
}

function dismissReconnect(){
  document.getElementById('reconnectBanner').style.display='none';
  if(_reconnectPoll){ clearInterval(_reconnectPoll); _reconnectPoll=null; }
  localStorage.removeItem('deeptrace_job_id');
}

function reconnectJob(){
  if(!_activeJobId) return;
  document.getElementById('reconnectBanner').style.display='none';

  const btn = document.getElementById('swapBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Reconnected – processing…';
  const prog = document.getElementById('swapProgress');
  const bar  = document.getElementById('swapBar');
  prog.style.display = 'block';
  showAlert('swapAlert','Reconnected to running job. Waiting for server to finish…','info');

  const jobId = _activeJobId;
  _reconnectPoll = setInterval(async()=>{
    try{
      const pr = await fetch('/api/progress/'+jobId);
      const pd = await pr.json();
      if(!pd.found){ clearInterval(_reconnectPoll); return; }

      if(pd.status === 'done'){
        clearInterval(_reconnectPoll);
        bar.style.width='100%';
        setTimeout(()=>prog.style.display='none',600);
        btn.disabled=false; btn.innerHTML='⚡ Swap Faces';
        showAlert('swapAlert','✅ Processing finished! Check the outputs folder or restart with a new swap.','success');
        localStorage.removeItem('deeptrace_job_id');
      } else if(pd.status === 'encoding'){
        bar.style.width='95%';
        btn.innerHTML='<span class="spinner"></span> Encoding H.264…';
      } else if(pd.total > 0){
        const pct = 20+Math.round(pd.done/pd.total*72);
        bar.style.width=pct+'%';
        const eta = pd.eta_seconds>0?(pd.eta_seconds<60?`${pd.eta_seconds}s`:`${Math.round(pd.eta_seconds/60)}m`):'';
        btn.innerHTML=`<span class="spinner"></span> Frame ${pd.done}/${pd.total} ${eta} · ${pd.fps_proc||0} fps`;
      }
    } catch(e){}
  }, 1000);
}

// Check on page load
window.addEventListener('load', ()=>{ setTimeout(checkForActiveJobs, 800); });

// ── Generic helpers ───────────────────────────────────────────────────────────
function showAlert(elId, msg, type){
  const el = document.getElementById(elId);
  el.innerHTML = msg;
  el.className = 'alert ' + type;
  el.style.display = 'block';
}
function hideAlert(elId){ document.getElementById(elId).style.display='none'; }

function previewFile(inputEl, previewEl){
  const file = inputEl.files[0];
  if(!file) return;
  const url = URL.createObjectURL(file);
  const el = document.getElementById(previewEl);
  el.src = url;
  el.style.display = 'block';
}

// ── Drag-and-drop upload zones ────────────────────────────────────────────────
['srcZone','tgtZone'].forEach(id=>{
  const zone = document.getElementById(id);
  if(!zone) return;
  zone.addEventListener('dragover', e=>{e.preventDefault(); zone.classList.add('drag');});
  zone.addEventListener('dragleave', ()=>zone.classList.remove('drag'));
  zone.addEventListener('drop', e=>{
    e.preventDefault(); zone.classList.remove('drag');
    const inp = zone.querySelector('input[type=file]');
    inp.files = e.dataTransfer.files;
    inp.dispatchEvent(new Event('change'));
  });
});

document.getElementById('srcFile').addEventListener('change', function(){
  previewFile(this, 'srcPreview');
});
document.getElementById('tgtFile').addEventListener('change', function(){
  const file = this.files[0];
  if(!file) return;
  const url = URL.createObjectURL(file);
  const isVideo = /\.(mp4|avi|mov|mkv)$/i.test(file.name) || file.type.startsWith('video/');

  // Show image or video preview in the target zone
  const prevZone = document.getElementById('tgtZone');
  prevZone.querySelectorAll('img.preview-img, video.preview-img').forEach(el=>el.remove());

  if(isVideo){
    // Video: disable enhancer by default (too slow per-frame), show video options
    document.getElementById('enhanceCheck').checked = false;
    document.getElementById('enhanceHint').textContent = 'OFF by default for video — enable only for short clips';
    const v = document.createElement('video');
    v.src = url; v.controls = true; v.muted = true; v.className = 'preview-img';
    v.style.cssText = 'display:block;max-width:100%;max-height:220px;border-radius:8px;margin-top:10px';
    v.addEventListener('loadedmetadata', ()=>{
      const estFrames = Math.round(v.duration * 30);
      document.getElementById('videoOpts').style.display = 'block';
      updateTimeEstimate(estFrames);
    });
    prevZone.appendChild(v);
  } else {
    // Image: enable enhancer by default (fast, big quality boost)
    document.getElementById('enhanceCheck').checked = true;
    document.getElementById('enhanceHint').textContent = 'ON by default for images — auto-disabled for video';
    document.getElementById('videoOpts').style.display = 'none';
    const img = document.getElementById('tgtPreview');
    img.src = url; img.style.display='block';
  }

  document.getElementById('swapOutput').style.display='none';
  hideAlert('swapAlert');
});

// Update live when settings change
['detectInterval','maxSide','enhanceCheck'].forEach(id=>{
  const el = document.getElementById(id);
  if(el) el.addEventListener('change', ()=>{
    const v = document.querySelector('#tgtZone video.preview-img');
    if(v && v.duration) updateTimeEstimate(Math.round(v.duration * 30));
  });
});

function updateTimeEstimate(frames){
  const enhance = document.getElementById('enhanceCheck').checked;
  const interval = parseInt(document.getElementById('detectInterval').value || '3');
  const maxSide  = parseInt(document.getElementById('maxSide').value || '720');

  // Per-frame cost model (empirical CPU estimates)
  const detectCost = 0.25;   // seconds per detection run
  const swapCost   = 0.65;   // seconds per swap (inswapper_128)
  const enhCost    = 3.0;    // seconds per enhance (gfpgan)
  const resizeFactor = maxSide > 0 ? Math.min(1.0, (maxSide / 1080) ** 1.5) : 1.0;

  const perFrame = (detectCost / interval + swapCost + (enhance ? enhCost : 0)) * resizeFactor;
  const totalSec = Math.round(frames * perFrame);

  let timeStr;
  if(totalSec < 60) timeStr = `~${totalSec}s`;
  else if(totalSec < 3600) timeStr = `~${Math.round(totalSec/60)} min`;
  else timeStr = `~${(totalSec/3600).toFixed(1)} hr`;

  const fps = (1/perFrame).toFixed(2);
  document.getElementById('timeEstimate').innerHTML =
    `⏱ Estimated: <strong style="color:var(--accent)">${timeStr}</strong> &nbsp;|&nbsp; `+
    `${frames} frames &nbsp;|&nbsp; ~${fps} frames/sec &nbsp;|&nbsp; `+
    `<span style="color:var(--warn)">Tip: disable enhancer + detect every 10 frames for max speed</span>`;
}

// ── Swap tab ──────────────────────────────────────────────────────────────────
function doSwap(){
  const src = document.getElementById('srcFile').files[0];
  const tgt = document.getElementById('tgtFile').files[0];
  if(!src || !tgt){ showAlert('swapAlert','Select both source and target files','warn'); return; }

  const isVideo = /\.(mp4|avi|mov|mkv)$/i.test(tgt.name) || tgt.type.startsWith('video/');
  const jobId = Math.random().toString(36).substr(2,9);

  const btn = document.getElementById('swapBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Uploading…';
  hideAlert('swapAlert');
  document.getElementById('swapOutput').style.display='none';

  const prog = document.getElementById('swapProgress');
  const bar  = document.getElementById('swapBar');
  prog.style.display = 'block';
  bar.style.width = '0%';

  if(isVideo){
    showAlert('swapAlert',
      '⏳ Video detected. Processing can take several minutes on CPU. Each frame is processed individually.',
      'info');
  }

  const fd = new FormData();
  fd.append('source', src);
  fd.append('target', tgt);
  fd.append('enhance', document.getElementById('enhanceCheck').checked ? '1' : '0');
  fd.append('pitch_semitones', document.getElementById('pitchRange').value);
  fd.append('job_id', jobId);
  if(isVideo){
    fd.append('detect_interval', document.getElementById('detectInterval').value || '5');
    fd.append('max_side', document.getElementById('maxSide').value || '720');
  }

  // Poll server for frame-by-frame progress + ETA
  let pollTimer = null;
  if(isVideo){
    pollTimer = setInterval(async()=>{
      try{
        const pr = await fetch('/api/progress/'+jobId);
        const pd = await pr.json();
        if(pd.found && pd.total > 0){
          if(pd.status === 'encoding'){
            bar.style.width = '95%';
            btn.innerHTML = '<span class="spinner"></span> Encoding H.264…';
          } else if(pd.status === 'processing'){
            const pct = 20 + Math.round((pd.done/pd.total)*72);
            bar.style.width = pct+'%';
            const eta = pd.eta_seconds > 0
              ? (pd.eta_seconds < 60 ? `${pd.eta_seconds}s left` : `${Math.round(pd.eta_seconds/60)}m left`)
              : '';
            const fps = pd.fps_proc > 0 ? ` · ${pd.fps_proc} fps` : '';
            btn.innerHTML = `<span class="spinner"></span> Frame ${pd.done}/${pd.total} ${eta}${fps}`;
          }
        }
      } catch(e){}
    }, 1000);
  }

  // Persist job so page refresh can reconnect
  if(isVideo) localStorage.setItem('deeptrace_job_id', jobId);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/swap');
  xhr.timeout = 3600000; // 1 hour for long videos

  xhr.upload.onprogress = (e)=>{
    if(!e.lengthComputable) return;
    const pct = Math.round((e.loaded/e.total) * (isVideo ? 15 : 80));
    bar.style.width = pct+'%';
    const mb = (e.loaded/1048576).toFixed(1);
    const tot = (e.total/1048576).toFixed(1);
    btn.innerHTML = `<span class="spinner"></span> Uploading ${mb}/${tot} MB`;
  };

  xhr.upload.onload = ()=>{
    if(isVideo){
      bar.style.width = '20%';
      btn.innerHTML = '<span class="spinner"></span> Processing frames…';
    } else {
      bar.style.width = '50%';
      btn.innerHTML = '<span class="spinner"></span> Swapping face…';
    }
  };

  xhr.onload = ()=>{
    if(pollTimer) clearInterval(pollTimer);
    localStorage.removeItem('deeptrace_job_id');
    bar.style.width = '100%';
    setTimeout(()=>{ prog.style.display='none'; bar.style.width='0%'; }, 800);
    btn.disabled=false; btn.innerHTML='⚡ Swap Faces';
    try{
      const data = JSON.parse(xhr.responseText);
      if(data.success){
        hideAlert('swapAlert');
        const extra = isVideo
          ? ` | ${data.frames_processed} frames @ ${data.fps_achieved} fps (${data.output_resolution||''})`
          : '';
        showAlert('swapAlert',`✅ Done in ${data.processing_time}s — ${data.faces_swapped} face(s) swapped${extra}`,'success');
        renderOutput(data.output_filename, data.output_type, data.preview_b64);
      } else {
        showAlert('swapAlert','❌ '+(data.error||'Swap failed'),'error');
      }
    } catch(e){ showAlert('swapAlert','Server error – check logs','error'); }
  };

  xhr.onerror = xhr.ontimeout = ()=>{
    if(pollTimer) clearInterval(pollTimer);
    prog.style.display='none';
    btn.disabled=false; btn.innerHTML='⚡ Swap Faces';
    localStorage.removeItem('deeptrace_job_id');
    showAlert('swapAlert','Network error or timeout. Check server logs.','error');
  };

  xhr.send(fd);
}

function renderOutput(filename, type, previewB64){
  const box = document.getElementById('swapOutput');
  box.style.display = 'block';
  const img = document.getElementById('outImg');
  const vid = document.getElementById('outVid');
  const dl  = document.getElementById('outDl');
  const url = '/output/' + filename;

  if(type==='video' || filename.match(/\.(mp4|avi|mov|mkv)$/i)){
    img.style.display='none';
    vid.style.display='block';
    vid.src = url;
    vid.load();
  } else {
    vid.style.display='none';
    img.style.display='block';

    if(previewB64){
      // Use embedded base64 — no extra HTTP request, no file-serving issues
      const mime = filename.toLowerCase().endsWith('.png') ? 'image/png' : 
                   (filename.toLowerCase().endsWith('.webp') ? 'image/webp' : 'image/jpeg');
      img.src = `data:${mime};base64,${previewB64}`;
    } else {
      // Fallback: fetch from /output/ route
      img.src = url + '?t=' + Date.now();
    }

    img.onerror = () => {
      img.style.display = 'none';
      showAlert('swapAlert',
        '⚠️ Swap succeeded. <a href="'+url+'" target="_blank" style="color:inherit;font-weight:700">Click here to open the image</a>', 'warn');
    };
  }

  // Download button always points to the served file (proper filename)
  dl.href = url;
  dl.download = filename;

  const dlHtml = document.getElementById('outDlHtml');
  if(dlHtml) {
    if(type === 'image' && previewB64){
      dlHtml.style.display = 'inline-block';
      const mime = filename.toLowerCase().endsWith('.png') ? 'image/png' : 
                   (filename.toLowerCase().endsWith('.webp') ? 'image/webp' : 'image/jpeg');
      const b64Src = `data:${mime};base64,${previewB64}`;
      const htmlContent = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>DeepTrace Swap Result</title><style>body{font-family:system-ui,sans-serif;background:#111;color:#fff;display:flex;flex-direction:column;align-items:center;padding:2rem;margin:0}.container{background:#222;padding:20px;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.5);max-width:90%;text-align:center}img{max-width:100%;border-radius:8px;margin-top:15px}h2{margin-top:0;color:#4ade80}</style></head><body><div class="container"><h2>DeepTrace Face Swap Result</h2><p>File: ${filename}</p><img src="${b64Src}" alt="Result Image"></div></body></html>`;
      const blob = new Blob([htmlContent], {type: 'text/html'});
      dlHtml.href = URL.createObjectURL(blob);
      dlHtml.download = filename.replace(/\.[^/.]+$/, "") + ".html";
    } else {
      dlHtml.style.display = 'none';
    }
  }

  // "Open in new tab" button
  let openLink = document.getElementById('outOpenLink');
  if(!openLink){
    openLink = document.createElement('a');
    openLink.id = 'outOpenLink';
    openLink.className = 'btn btn-outline btn-sm';
    openLink.style.marginLeft = '8px';
    openLink.textContent = '↗ Open in new tab';
    openLink.target = '_blank';
    dl.parentNode.appendChild(openLink);
  }
  openLink.href = url;
}

function resetSwap(){
  ['srcFile','tgtFile'].forEach(id=>document.getElementById(id).value='');
  ['srcPreview','tgtPreview'].forEach(id=>{const e=document.getElementById(id);e.src='';e.style.display='none';});
  document.getElementById('swapOutput').style.display='none';
  hideAlert('swapAlert');
}

// ── Webcam tab ────────────────────────────────────────────────────────────────
let camStream = null;
let camRunning = false;
let camLoop = null;

document.getElementById('wcSrcFile').addEventListener('change', async function(){
  const file = this.files[0];
  if(!file) return;
  const fd = new FormData();
  fd.append('source', file);
  showAlert('wcAlert','Uploading source face…','info');
  try{
    const res = await fetch('/api/webcam/set-source',{method:'POST',body:fd});
    const data = await res.json();
    if(data.success){
      hideAlert('wcAlert');
      const thumb = document.getElementById('srcPreviewThumb');
      thumb.src = 'data:image/jpeg;base64,' + data.preview_b64;
      thumb.style.display='block';
      document.getElementById('wcSrcStatus').textContent = `Ready (score ${data.score.toFixed(2)})`;
    } else {
      showAlert('wcAlert', '❌ ' + data.error, 'error');
    }
  } catch(e){ showAlert('wcAlert','Network error: '+e.message,'error'); }
});

async function toggleCamera(){
  if(camRunning){ stopWebcam(); return; }

  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showAlert('wcAlert', '📷 Camera access requires a Secure Context.<br>Please access the app via <b>localhost</b> (http://localhost:5000) or use HTTPS.<br><span style="font-size:0.85em;color:var(--muted)">Modern browsers block camera access on IP addresses like 192.168.x.x for security.</span>', 'error');
    return;
  }

  try{
    camStream = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}});
    const vid = document.getElementById('webcamVideo');
    vid.srcObject = camStream;
    camRunning = true;
    document.getElementById('camToggleBtn').textContent='⏸ Pause';
    document.getElementById('camStopBtn').style.display='inline-block';
    startCamLoop();
  } catch(e){
    showAlert('wcAlert','Camera access denied: '+e.message,'error');
  }
}

function stopWebcam(){
  camRunning=false;
  if(camLoop){ clearTimeout(camLoop); camLoop=null; }
  if(camStream){ camStream.getTracks().forEach(t=>t.stop()); camStream=null; }
  document.getElementById('webcamVideo').srcObject=null;
  document.getElementById('camToggleBtn').textContent='▶ Start Camera';
  document.getElementById('camStopBtn').style.display='none';
}

async function startCamLoop(){
  if(!camRunning) return;
  const canvas = document.getElementById('captureCanvas');
  const vid = document.getElementById('webcamVideo');
  canvas.width=vid.videoWidth||640;
  canvas.height=vid.videoHeight||480;
  const ctx=canvas.getContext('2d');
  ctx.drawImage(vid,0,0);
  const b64=canvas.toDataURL('image/jpeg',0.7).split(',')[1];

  try{
    const res=await fetch('/api/webcam/process-frame',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({frame_b64:b64, enhance:document.getElementById('wcEnhance').checked})
    });
    const data=await res.json();
    if(data.success){
      document.getElementById('webcamResult').src='data:image/jpeg;base64,'+data.result_b64;
      document.getElementById('fpsBadge').textContent=data.fps+' fps';
      document.getElementById('facesBadge').textContent=data.faces_found+' face(s)';
    }
  } catch(e){ /* silently retry */ }

  camLoop = setTimeout(startCamLoop, 50); // ~20 fps attempt; server limits actual rate
}

// ── Multi-face tab ────────────────────────────────────────────────────────────
let mfSelectedIndices = new Set();
let mfFaceCount = 0;
let mfTgtPath = null;

document.getElementById('mfSrcFile').addEventListener('change',function(){previewFile(this,'mfSrcPreview');});

document.getElementById('mfTgtFile').addEventListener('change', async function(){
  previewFile(this,'mfTgtPreview');
  const file=this.files[0];
  if(!file) return;
  const fd=new FormData();
  fd.append('image',file);
  showAlert('mfAlert','Detecting faces…','info');
  try{
    const res=await fetch('/api/detect-faces',{method:'POST',body:fd});
    const data=await res.json();
    if(data.success){
      hideAlert('mfAlert');
      mfTgtPath=data.image_path;
      mfFaceCount=data.count;
      mfSelectedIndices.clear();
      renderFaceGrid(data.faces);
    } else {
      showAlert('mfAlert','❌ '+data.error,'error');
      document.getElementById('mfFaceGrid').innerHTML='<span style="color:var(--muted);font-size:.85rem">No faces found</span>';
    }
  } catch(e){ showAlert('mfAlert','Error: '+e.message,'error'); }
});

function renderFaceGrid(faces){
  const grid=document.getElementById('mfFaceGrid');
  if(!faces.length){ grid.innerHTML='<span style="color:var(--muted);font-size:.85rem">No faces detected</span>'; return; }
  grid.innerHTML='';
  faces.forEach((f,i)=>{
    const img=document.createElement('img');
    img.src='data:image/jpeg;base64,'+f.b64;
    img.className='face-thumb';
    img.title=`Face ${i+1} (score ${f.score.toFixed(2)})`;
    img.onclick=()=>toggleFace(i,img);
    grid.appendChild(img);
  });
  updateSelInfo();
}

function toggleFace(i, el){
  if(mfSelectedIndices.has(i)){ mfSelectedIndices.delete(i); el.classList.remove('selected'); }
  else { mfSelectedIndices.add(i); el.classList.add('selected'); }
  updateSelInfo();
}
function selectAllFaces(){
  document.querySelectorAll('.face-thumb').forEach((el,i)=>{el.classList.add('selected'); mfSelectedIndices.add(i);});
  updateSelInfo();
}
function deselectAllFaces(){
  document.querySelectorAll('.face-thumb').forEach(el=>el.classList.remove('selected'));
  mfSelectedIndices.clear();
  updateSelInfo();
}
function updateSelInfo(){
  const n=mfSelectedIndices.size;
  document.getElementById('mfSelInfo').textContent=
    n===0 ? `0 selected → all ${mfFaceCount} face(s) will be swapped`
          : `${n} face(s) selected`;
}

async function doMultiSwap(){
  const src=document.getElementById('mfSrcFile').files[0];
  const tgt=document.getElementById('mfTgtFile').files[0];
  if(!src||!tgt){ showAlert('mfAlert','Upload both source and target images','warn'); return; }

  const btn=document.getElementById('mfSwapBtn');
  btn.disabled=true; btn.innerHTML='<span class="spinner"></span> Processing…';
  hideAlert('mfAlert');

  const fd=new FormData();
  fd.append('source',src);
  fd.append('target',tgt);
  const indices=mfSelectedIndices.size>0?JSON.stringify([...mfSelectedIndices]):'[]';
  fd.append('face_indices',indices);
  fd.append('enhance','1');

  try{
    const res=await fetch('/api/swap',{method:'POST',body:fd});
    const data=await res.json();
    if(data.success){
      showAlert('mfAlert',`✅ Done – ${data.faces_swapped} face(s) swapped in ${data.processing_time}s`,'success');
      const box=document.getElementById('mfOutput');
      box.style.display='block';
      const url='/output/'+data.output_filename;
      const mfImg = document.getElementById('mfOutImg');
      if(data.preview_b64){
        const mime = data.output_filename.toLowerCase().endsWith('.png') ? 'image/png' : 
                     (data.output_filename.toLowerCase().endsWith('.webp') ? 'image/webp' : 'image/jpeg');
        mfImg.src = `data:${mime};base64,${data.preview_b64}`;
      } else {
        mfImg.src = url + '?t=' + Date.now();
      }
      document.getElementById('mfOutDl').href=url;
      document.getElementById('mfOutDl').download=data.output_filename;

      const mfDlHtml = document.getElementById('mfOutDlHtml');
      if(mfDlHtml) {
        if(data.preview_b64) {
          mfDlHtml.style.display = 'inline-block';
          const mime = data.output_filename.toLowerCase().endsWith('.png') ? 'image/png' : 
                       (data.output_filename.toLowerCase().endsWith('.webp') ? 'image/webp' : 'image/jpeg');
          const b64Src = `data:${mime};base64,${data.preview_b64}`;
          const htmlContent = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>DeepTrace Swap Result</title><style>body{font-family:system-ui,sans-serif;background:#111;color:#fff;display:flex;flex-direction:column;align-items:center;padding:2rem;margin:0}.container{background:#222;padding:20px;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.5);max-width:90%;text-align:center}img{max-width:100%;border-radius:8px;margin-top:15px}h2{margin-top:0;color:#4ade80}</style></head><body><div class="container"><h2>DeepTrace Face Swap Result</h2><p>File: ${data.output_filename}</p><img src="${b64Src}" alt="Result Image"></div></body></html>`;
          const blob = new Blob([htmlContent], {type: 'text/html'});
          mfDlHtml.href = URL.createObjectURL(blob);
          mfDlHtml.download = data.output_filename.replace(/\.[^/.]+$/, "") + ".html";
        } else {
          mfDlHtml.style.display = 'none';
        }
      }
      document.getElementById('mfOutMeta').innerHTML=
        `<span class="stat">⏱ ${data.processing_time}s</span>
         <span class="stat">👤 ${data.faces_swapped} face(s)</span>`;
    } else {
      showAlert('mfAlert','❌ '+(data.error||'Swap failed'),'error');
    }
  } catch(e){ showAlert('mfAlert','Network error: '+e.message,'error'); }
  finally { btn.disabled=false; btn.innerHTML='⚡ Swap Selected Faces'; }
}
</script>
</body>
</html>
"""


if __name__ == '__main__':
    if initialize_deeptrace():
        _INIT_OK = True
        # Warm up models in background so first request is fast
        threading.Thread(target=warmup_models, daemon=True).start()
        logger.info('Starting DeepTrace v2 on http://0.0.0.0:5000')
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        logger.error('Initialisation failed – exiting')
