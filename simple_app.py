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
from deeptrace import state_manager, face_detector, face_analyser, face_masker
from deeptrace.processors.modules.face_swapper import core as face_swapper
import deeptrace.processors.modules.face_enhancer.core as face_enhancer
from deeptrace.vision import read_static_image, write_image
from deeptrace.filesystem import is_image, is_video
from deeptrace.execution import get_available_execution_providers

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

# ── Quality tuning constants ───────────────────────────────────────────────────
# Cross-gender realistic swap: transfer the facial skin + identity-bearing features
# (skin, eyes, eyebrows, nose) for a strong, accurate likeness, but DELIBERATELY
# exclude mouth/upper-lip/lower-lip — those regions carry the source's mustache and
# lip-hair, and compositing them is what painted phantom facial hair onto
# opposite-gender targets. The target keeps its own mouth/lips, so a female target
# stays beardless while still receiving the source's full upper-face identity.
CROSS_GENDER_REGIONS = ['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'nose']

# Faithful cross-gender: full identity transfer (incl. skin/jaw/beard, by design) but
# STILL omit mouth/upper-lip/lower-lip so a female source's lips/lipstick are not painted
# onto a male target (the "feminine pink lips on a man" uncanny artefact). The target
# keeps its own lips; everything else gets the source identity at full strength.
FAITHFUL_CROSS_GENDER_REGIONS = ['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'nose', 'glasses']

# Lower-face padding (% of crop height) applied to the BOX mask in cross-gender mode.
# It crops the chin/jaw — the source male's densest beard zone — out of the swap so
# that area is never composited onto a female target, while the box mask's feathered
# edge keeps the transition soft (no hard "jawline" seam). The target keeps its own
# clean chin. Combined with CROSS_GENDER_REGIONS (which omits mouth/upper-lip/lower-lip)
# this removes both the mustache and the chin-beard blotching that made opposite-gender
# swaps look pasted-on. See swap_frame()/swap_frame_multiple().
CROSS_GENDER_BOTTOM_PAD = 10

# Embedding blend + enhancer fidelity for cross-gender swaps.
# 0.50 = pure source identity (interp maps to 0.0 scalar). The earlier 0.35 blended ~10%
# of the TARGET back in which, combined with the limited region mask, made some swaps
# barely differ from the target ("same as input"). With the high-quality hyperswap model
# the clean-skin / no-beard look no longer needs an aggressive low enhancer fidelity, so
# we keep CodeFormer at 0.70 (same as same-gender) to preserve the swapped identity
# detail instead of regenerating it toward a generic face. Validated via _fix_test.py.
CROSS_GENDER_SWAPPER_WEIGHT = 0.50
CROSS_GENDER_ENHANCER_WEIGHT = 0.70

# Active enhancer model, resolved at warm-up to a model whose weights are ALREADY
# on disk. None ⇒ no enhancer available ⇒ the pipeline runs swap-only (still works).
# We never trigger a multi-hundred-MB download on a request thread — that is what
# made the first swap appear to "hang with no output". CodeFormer is preferred when
# present (gentler on facial hair than GFPGAN); otherwise we use whatever exists.
_ENHANCER_ACTIVE: str | None = None

# Preference order: CodeFormer first (least beard hallucination), then GFPGAN, etc.
_ENHANCER_PREFERENCE = [
    'codeformer', 'gfpgan_1.4', 'gfpgan_1.3', 'gfpgan_1.2',
    'gpen_bfr_512', 'gpen_bfr_256', 'restoreformer_plus_plus',
]


def _image_pixel_boost(preferred: str = '256x256') -> str:
    """Return a pixel-boost resolution valid for the active swapper model.

    Pixel-boost runs the swapper on N×N tiles to recover detail beyond the model's
    native crop. Each model exposes a different set of valid resolutions, so we pick
    `preferred` when the current model supports it, otherwise fall back to that model's
    smallest option (keeps us correct if the default model is ever changed).
    """
    from deeptrace.processors.modules.face_swapper import choices as _fs_choices
    model = state_manager.get_item('face_swapper_model')
    options = _fs_choices.face_swapper_set.get(model, ['256x256'])
    return preferred if preferred in options else options[0]


def _enhancer_model_present(name: str) -> bool:
    """True if the enhancer ONNX weights for `name` already exist locally."""
    path = os.path.join(_SCRIPT_DIR, '.assets', 'models', f'{name}.onnx')
    return os.path.isfile(path) and os.path.getsize(path) > 1_000_000


def _resolve_available_enhancer() -> str | None:
    """Pick the best enhancer whose weights are already downloaded; never downloads."""
    for name in _ENHANCER_PREFERENCE:
        if _enhancer_model_present(name):
            return name
    return None


# Swapper preference: native 256px hyperswap (far more realistic than 128px inswapper)
# when its weights are on disk, otherwise fall back to inswapper so the app always runs.
# Download hyperswap with `python download_hyperswap.py`.
_SWAPPER_PREFERENCE = ['hyperswap_1a_256', 'inswapper_128_fp16']


def _resolve_available_swapper() -> str:
    """Best swapper model whose weights already exist on disk (never downloads)."""
    for name in _SWAPPER_PREFERENCE:
        path = os.path.join(_SCRIPT_DIR, '.assets', 'models', f'{name}.onnx')
        if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
            return name
    return 'inswapper_128_fp16'


def _apply_quality_profile(realism_mode: str) -> str:
    """Map a UI-friendly realism_mode to swap behaviour + enhancer tuning.

    Returns the internal `gender_match_mode` to use:
      - 'realistic' → 'features_only'  (cross-gender = inner-feature swap, no beard
                                        transfer; same-gender = full box swap)
      - 'faithful'  → 'faithful'       (full identity transfer for every face,
                                        incl. the source's facial hair — by design)
    """
    if _ENHANCER_ACTIVE:
        state_manager.set_item('face_enhancer_model', _ENHANCER_ACTIVE)
    if realism_mode == 'faithful':
        # Preserve the source identity as-is; keep enhancer gentle so it doesn't
        # over-sharpen, but high fidelity so it doesn't invent texture.
        state_manager.set_item('face_enhancer_blend', 85)
        state_manager.set_item('face_enhancer_weight', 0.85)
        return 'faithful'
    # realistic (default). 0.70 fidelity (was 0.80) lets CodeFormer regenerate slightly
    # cleaner, more natural skin instead of faithfully reproducing 128px swap artefacts —
    # the single biggest "looks real" lever for SAME-gender swaps. Cross-gender faces
    # drop further to CROSS_GENDER_ENHANCER_WEIGHT (0.5) inside swap_frame().
    state_manager.set_item('face_enhancer_blend', 75)   # was 100 → less hallucinated stubble
    state_manager.set_item('face_enhancer_weight', 0.70)
    return 'features_only'


def _resolve_swap_mode(form) -> str:
    """Resolve the request's swap mode into an internal gender_match_mode and
    apply the matching enhancer profile.

    Preferred param: realism_mode = 'realistic' | 'faithful'.
    Back-compat: a legacy explicit gender_match_mode still wins if provided.
    """
    legacy = (form.get('gender_match_mode') or '').strip()
    if legacy:
        _apply_quality_profile('faithful' if legacy == 'faithful' else 'realistic')
        return legacy
    realism = (form.get('realism_mode') or 'realistic').strip() or 'realistic'
    return _apply_quality_profile(realism)


def check_bisenet_model() -> bool:
    dest = Path(_SCRIPT_DIR) / '.assets' / 'models' / 'bisenet_resnet_34.onnx'
    if dest.exists() and dest.stat().st_size > 10000000:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    urls = [
        'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.onnx',
        'https://huggingface.co/facefusion/models/resolve/main/bisenet_resnet_34.onnx'
    ]
    import urllib.request
    for url in urls:
        try:
            logger.info(f"Downloading Bisenet model from {url}...")
            urllib.request.urlretrieve(url, str(dest))
            if dest.exists() and dest.stat().st_size > 10000000:
                logger.info("Bisenet model downloaded successfully.")
                return True
        except Exception as e:
            logger.warning(f"Failed to download Bisenet model from {url}: {e}")
            if dest.exists():
                try:
                    dest.unlink()
                except Exception:
                    pass
    return False


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
        available_providers = get_available_execution_providers()
        logger.info(f"Detected available execution providers: {available_providers}")
        state_manager.init_item('execution_providers', available_providers)
        state_manager.init_item('execution_device_ids', [0])
        state_manager.init_item('execution_thread_count', 4)

        # Face detector  ← BUG FIX: was 0.3, raised false-negative rate
        state_manager.init_item('face_detector_model', 'yolo_face')
        state_manager.init_item('face_detector_size', '640x640')
        state_manager.init_item('face_detector_score', 0.15)   # lowered from 0.3
        state_manager.init_item('face_detector_angles', [0])
        state_manager.init_item('face_detector_margin', [0, 0, 0, 0])

        # Face swapper
        # Prefer the native 256px hyperswap model for hyper-realism; gracefully fall back
        # to inswapper_128_fp16 if hyperswap weights aren't downloaded yet.
        _swapper = _resolve_available_swapper()
        state_manager.init_item('face_swapper_model', _swapper)
        logger.info(f'Face swapper model: {_swapper}')
        # Pixel-boost runs the swapper on tiles to recover real facial detail the bare
        # 128px output never had. 256x256 (2x2) is the sweet spot: noticeably sharper
        # skin/eyes with no visible tile seams. Overridden per-job (256 for images,
        # 128 for video where speed matters). See process_image_swap().
        state_manager.init_item('face_swapper_pixel_boost', '256x256')
        state_manager.init_item('face_swapper_weight', 0.5)             # 0.5 = pure source identity (interp maps to 0.0 scalar)

        # Face enhancer — the actual model is resolved at warm-up to whatever weights
        # are already on disk (see _resolve_available_enhancer); enhancement is skipped
        # entirely if none are present. blend < 100 so the enhancer can't fully repaint
        # the face with invented stubble. Per-request tuning in _apply_quality_profile().
        state_manager.init_item('face_enhancer_model', 'gfpgan_1.4')
        state_manager.init_item('face_enhancer_blend', 75)   # was 100 → less hallucinated beard
        state_manager.init_item('face_enhancer_weight', 0.80)  # high fidelity = minimal invention

        # Recogniser / landmarker
        state_manager.init_item('face_recognizer_model', 'arcface_inswapper')
        state_manager.init_item('face_landmarker_model', '2dfan4')
        state_manager.init_item('face_landmarker_score', 0.3)

        # Masks — softer blur + padding captures full face boundary cleanly
        state_manager.init_item('face_mask_types', ['box'])
        state_manager.init_item('face_mask_blur', 0.3)              # tight boundary — old 0.6 bled into ears/lips
        state_manager.init_item('face_mask_padding', [0, 0, 0, 0])  # zero padding — let blur_area alone define the soft edge
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
    global _WARMUP_DONE, _ENHANCER_ACTIVE
    # Resolve the enhancer to a model that is ALREADY on disk — never download here
    # (a blocking download is what made the first swap appear to hang).
    _ENHANCER_ACTIVE = _resolve_available_enhancer()
    if _ENHANCER_ACTIVE:
        state_manager.set_item('face_enhancer_model', _ENHANCER_ACTIVE)
        logger.info(f'Face enhancer active: {_ENHANCER_ACTIVE}')
    else:
        logger.warning('No face-enhancer weights found on disk — running SWAP-ONLY '
                       '(no enhancement). Run "python download_enhancer.py" to add one '
                       '(codeformer recommended).')
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
                         pad_x_frac: float = 0.08,
                         pad_top_frac: float = 0.10,
                         pad_bot_frac: float = 0.12) -> np.ndarray:
    """
    Elliptical, Gaussian-feathered face mask sized to the detected bounding box.
    Keep pads tight — the blend handles the transition; large pads bleed into neck/ears.
    """
    x1, y1, x2, y2 = [int(v) for v in face.bounding_box]
    fw, fh = x2 - x1, y2 - y1

    px  = max(8,  int(fw * pad_x_frac))
    pt  = max(6,  int(fh * pad_top_frac))
    pb  = max(10, int(fh * pad_bot_frac))

    x1e = max(0, x1 - px);  y1e = max(0, y1 - pt)
    x2e = min(w, x2 + px);  y2e = min(h, y2 + pb)

    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = (x1e + x2e) // 2, (y1e + y2e) // 2
    rx, ry = (x2e - x1e) // 2, (y2e - y1e) // 2
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    # Feather: proportional to face size but kept moderate
    k = max(11, (min(rx, ry) // 8) * 2 + 1)
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
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    inner  = cv2.erode(mask, k, iterations=1)
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


def _unsharp_face(frame: np.ndarray, mask: np.ndarray,
                  amount: float = 0.55, radius: float = 1.2) -> np.ndarray:
    """Unsharp-mask the face region only, to restore the micro-detail (pores, brow
    hairs, lash/iris edges) that the 256/512px swap and CodeFormer smoothing wash out.

    The swapped face is the only soft part of the frame — the background, hair and
    beard are the original sharp pixels — so we confine sharpening to the feathered
    ellipse `mask` (0..255). This is what removes the "blurry face, sharp surroundings"
    look without over-sharpening the rest of the image. Validated by _sharp_test.py.
    """
    blur = cv2.GaussianBlur(frame, (0, 0), radius)
    sharp = cv2.addWeighted(frame, 1.0 + amount, blur, -amount, 0)
    m = (mask.astype(np.float32) / 255.0)[..., None]
    return (sharp * m + frame.astype(np.float32) * (1.0 - m)).astype(np.uint8)


def _restore_target_mouth(result: np.ndarray, original: np.ndarray, face,
                          grow: float = 1.35, feather: float = 9.0) -> np.ndarray:
    """Blend the target's ORIGINAL mouth/lips back into a cross-gender swap.

    Cross-gender masking already excludes the lip regions, but the parser mask's
    feathered edge still lets part of the source mouth bleed through — e.g. a smiling
    female source's parted/fuller lips landing on a closed-mouth male target (the "lip
    overlay" artefact). We rebuild a feathered mask from the 68-pt mouth landmarks
    (48..67), grown slightly past the lips, and composite the untouched target mouth
    over the result. The target keeps its own mouth shape/expression; only cross-gender
    swaps call this (same-gender swaps want the source mouth).
    """
    lm = face.landmark_set.get('68') if getattr(face, 'landmark_set', None) else None
    if lm is None or len(lm) < 68:
        return result
    pts = np.asarray(lm[48:68], dtype=np.float32)
    centre = pts.mean(axis=0)
    pts = ((pts - centre) * grow + centre).astype(np.int32)
    mask = np.zeros(result.shape[:2], dtype=np.float32)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), feather)[..., None]
    blended = original.astype(np.float32) * mask + result.astype(np.float32) * (1.0 - mask)
    return blended.astype(np.uint8)


def _lower_face_mask(shape, face, raise_to: float = 0.55):
    """Feathered mask over the target's lower face — jaw, chin, mouth and lower cheeks.

    Built from the 68-pt jaw contour (0..16) closed off by a horizontal line at
    `raise_to` of the brow→chin distance (~just under the nose). It therefore follows
    each face's real jaw/beard outline at any size/angle. Feather scales with face size.
    """
    lm = face.landmark_set.get('68') if getattr(face, 'landmark_set', None) else None
    if lm is None or len(lm) < 68:
        return None
    lm = np.asarray(lm, dtype=np.float32)
    jaw = lm[0:17]
    brow_y = lm[17:27, 1].mean()
    chin_y = lm[8, 1]
    top_y = brow_y + (chin_y - brow_y) * raise_to
    left_x, right_x = jaw[:, 0].min(), jaw[:, 0].max()
    poly = np.vstack([jaw, [[right_x, top_y], [left_x, top_y]]]).astype(np.int32)
    mask = np.zeros(shape[:2], dtype=np.float32)
    cv2.fillConvexPoly(mask, cv2.convexHull(poly), 1.0)
    feather = float(np.clip((right_x - left_x) * 0.04, 6.0, 22.0))
    return cv2.GaussianBlur(mask, (0, 0), feather)[..., None]


def _restore_lower_face(result: np.ndarray, original: np.ndarray, face) -> np.ndarray:
    """Composite the target's whole lower face back into a cross-gender swap.

    The cross-gender swap is run at full strength (full-face box mask) for maximum
    identity, then this keeps the TARGET's jaw/chin/mouth/beard. That simultaneously:
      • M→F: removes any beard/stubble the male source would have left on the chin/jaw,
      • F→M: preserves the male target's full beard as one consistent region (no patch),
      • both: keeps the target's own mouth shape (no lipstick / lip-overlay artefact).
    Identity still transfers strongly through the eyes, brows, nose, forehead and upper
    cheeks (the most identity-bearing regions). Replaces the old region-mask + chin-crop
    + mouth-restore combo, which limited identity and left a patchy beard. See _lf_test.py.
    """
    m = _lower_face_mask(result.shape, face)
    if m is None:
        return result
    blended = original.astype(np.float32) * m + result.astype(np.float32) * (1.0 - m)
    return blended.astype(np.uint8)


def _advanced_face_blend(swapped: np.ndarray, original: np.ndarray, faces) -> np.ndarray:
    """
    Post-swap finishing pass: LAB boundary colour transfer + face-region sharpening.

    LAB transfer closes any remaining skin-tone gap at the edge ring without
    re-introducing the original pixels (the pyramid blend was removed earlier because
    it re-blended with the original and caused a double-face artefact). The unsharp
    pass then restores micro-detail the swap/enhancer smoothed, fixing the soft face.
    """
    if not faces:
        return swapped

    h, w   = original.shape[:2]
    result = swapped.copy()

    for face in faces:
        try:
            mask = _build_ellipse_mask(h, w, face)
            # 0.40 (was 0.25): pulls the swapped skin tone closer to the target so the
            # boundary where the swapped cheek meets the target's own skin/beard stops
            # showing a visible tone seam — most noticeable on cross-gender swaps.
            result = _lab_colour_transfer(result, original, mask, strength=0.40)
            # Crisp the (soft) swapped face back up; background/hair stay untouched.
            result = _unsharp_face(result, mask)
        except Exception as exc:
            logger.warning(f'Colour blend/sharpen failed ({exc}) — skipping')

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


def _face_area(face) -> float:
    bb = face.bounding_box
    return float((bb[2] - bb[0]) * (bb[3] - bb[1]))


def pick_source_face(faces):
    """Choose the SOURCE identity face from a (left→right sorted) detection list.

    A source portrait is the subject of the photo — i.e. the LARGEST, clearest
    face. The old code took faces[0] (leftmost), which on a group/selfie photo
    grabbed a tiny, barely-detected background person whose garbage embedding
    made the swap melt into a rainbow blob. We pick the biggest face and, when
    several are comparably large, break ties toward the highest detector score.
    Returns None for an empty list.
    """
    if not faces:
        return None
    if len(faces) == 1:
        return faces[0]
    max_area = max(_face_area(f) for f in faces)
    # Among faces within 70% of the largest area, prefer the most confident one.
    contenders = [f for f in faces if _face_area(f) >= 0.7 * max_area]
    return max(contenders, key=lambda f: f.score_set.get('detector', 0.0))


def swap_frame(source_face, target_faces, frame: np.ndarray,
               face_indices=None, enhance: bool = True,
               gender_match_mode: str = 'features_only',
               stats: dict | None = None) -> np.ndarray:
    """
    Apply `source_face` onto selected `target_faces` in `frame`.
    `face_indices`: list of int indices into `target_faces` to swap (None = all).
    `stats`: optional dict; when provided it is populated with honest accounting
             ('attempted', 'swapped', 'skipped_gender', 'errors') so callers can
             tell whether the swap actually changed anything instead of silently
             returning the untouched target frame.
    """
    result = frame.copy()

    # Save default masking states to prevent thread race contamination
    orig_mask_types = state_manager.get_item('face_mask_types') or ['box']
    orig_mask_regions = state_manager.get_item('face_mask_regions') or []
    orig_mask_blur = state_manager.get_item('face_mask_blur') or 0.3
    orig_mask_padding = state_manager.get_item('face_mask_padding') or [0, 0, 0, 0]
    orig_swapper_weight = state_manager.get_item('face_swapper_weight')
    orig_enhancer_weight = state_manager.get_item('face_enhancer_weight')

    for idx, tface in enumerate(target_faces):
        if face_indices is not None and idx not in face_indices:
            continue

        if stats is not None:
            stats['attempted'] = stats.get('attempted', 0) + 1

        # Gender is used only for the optional 'match' skip mode. Every other mode does a
        # full-strength swap: identity-embedding tests proved that ANY cross-gender
        # region/lower-face restore collapsed the likeness (the output stayed closer to the
        # target than the source), whereas a clean full box swap with hyperswap transfers
        # strongly (d_src ~0.25–0.35, same as same-gender) AND looks realistic. A face swap
        # reproduces the source's whole identity incl. their gender traits — by design.
        source_gender = getattr(source_face, 'gender', None)
        target_gender = getattr(tface, 'gender', None)
        is_cross_gender = bool(source_gender and target_gender and source_gender != target_gender)

        if gender_match_mode == 'match' and is_cross_gender:
            logger.info(f'Skipping swap for face {idx} due to gender mismatch ({source_gender} -> {target_gender})')
            if stats is not None:
                stats['skipped_gender'] = stats.get('skipped_gender', 0) + 1
            continue

        try:
            # Full-face box swap at full source identity for every face.
            state_manager.set_item('face_mask_types', ['box'])
            state_manager.set_item('face_mask_blur', 0.3)
            state_manager.set_item('face_mask_padding', [0, 0, 0, 0])
            state_manager.set_item('face_swapper_weight', orig_swapper_weight)
            state_manager.set_item('face_enhancer_weight', orig_enhancer_weight)

            result = face_swapper.swap_face(source_face, tface, result)
            if enhance and _ENHANCER_ACTIVE:
                try:
                    result = face_enhancer.enhance_face(tface, result)
                except Exception as ee:
                    # Enhancement is best-effort: never let it discard a good swap.
                    logger.warning(f'enhance failed for face {idx} (kept swap): {ee}')
            if stats is not None:
                stats['swapped'] = stats.get('swapped', 0) + 1
        except Exception as e:
            logger.warning(f'swap/enhance failed for face {idx}: {e}\n{traceback.format_exc()}')
            if stats is not None:
                stats.setdefault('errors', []).append(f'face {idx}: {e}')

    # Restore original states
    state_manager.set_item('face_mask_types', orig_mask_types)
    state_manager.set_item('face_mask_regions', orig_mask_regions)
    state_manager.set_item('face_mask_blur', orig_mask_blur)
    state_manager.set_item('face_mask_padding', orig_mask_padding)
    state_manager.set_item('face_swapper_weight', orig_swapper_weight)
    state_manager.set_item('face_enhancer_weight', orig_enhancer_weight)

    return result


def _resize_frame(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max_side <= 0 or max(w, h) <= max_side:
        return frame
    scale = max_side / max(w, h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def process_image_swap(source_path: str, target_path: str, output_path: str,
                       face_indices=None, enhance: bool = True,
                       pitch_semitones: float = 0.0,
                       job_id: str = '',
                       detect_interval: int = 5,
                       max_side: int = 720,
                       gender_match_mode: str = 'features_only') -> dict:
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
    source_face = pick_source_face(source_faces)
    logger.info(f'Source face: {len(source_faces)} detected, using largest '
                f'(area {_face_area(source_face):.0f}, score {source_face.score_set.get("detector"):.3f})')

    # ── Image swap ───────────────────────────────────────────────────────────
    if is_image(target_path):
        # Images: 512x512 pixel-boost so the swapped face carries real detail instead of
        # a 256px crop stretched over a large face (the main cause of the "blurry face,
        # sharp surroundings" look). Speed is not a concern for a single frame.
        state_manager.set_item('face_swapper_pixel_boost', _image_pixel_boost('512x512'))
        target_frame = read_frame(target_path)
        if target_frame is None:
            return {'success': False, 'error': 'Cannot read target image'}

        target_faces = detect_faces_in_frame(target_frame)
        if not target_faces:
            return {'success': False, 'error': 'No face detected in the target image'}

        logger.info(f'{len(target_faces)} target face(s) detected')
        stats: dict = {}
        result = swap_frame(source_face, target_faces, target_frame,
                            face_indices=face_indices, enhance=enhance,
                            gender_match_mode=gender_match_mode, stats=stats)

        # Honest failure: if nothing was actually swapped, do NOT return the
        # untouched target dressed up as success (the old silent-fallback bug).
        if stats.get('swapped', 0) == 0:
            if stats.get('skipped_gender', 0) > 0:
                return {'success': False, 'error':
                        'All target faces were skipped because their gender did not match the '
                        'source. Switch to "realistic" or "faithful" mode to swap across genders.'}
            errs = '; '.join(stats.get('errors', [])) or 'unknown error'
            return {'success': False, 'error': f'Face swap failed — no faces were swapped ({errs}).'}

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
            'faces_swapped': stats.get('swapped', 0),
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
        # Video: request the smallest valid boost — speed matters more per frame
        # (inswapper→128, hyperswap→256 since it has no 128 option).
        state_manager.set_item('face_swapper_pixel_boost', _image_pixel_boost('128x128'))

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

        # ThreadPoolExecutor to run swap_frame in parallel
        # Get thread count from state manager
        max_workers = max(1, int(state_manager.get_item('execution_thread_count') or 4))
        logger.info(f"Using parallel in-memory video processing with {max_workers} worker threads")

        futures = {}
        processed_frames = {}
        frame_idx = 0
        write_idx = 0
        
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while True:
                # Limit memory consumption by back-pressuring the reader if too many frames are in flight
                while len(futures) >= max_workers * 2:
                    time.sleep(0.005)
                    # Check for completed futures
                    completed_ids = [fid for fid, fut in futures.items() if fut.done()]
                    for fid in completed_ids:
                        try:
                            processed_frames[fid] = futures[fid].result()
                        except Exception as fut_err:
                            logger.error(f"Failed to process frame {fid}: {fut_err}")
                            skip_count += 1
                        del futures[fid]

                ok, frame = cap.read()
                if not ok:
                    break
                frame = _resize_frame(frame, max_side)

                # Re-detect faces sequentially in the reader thread to avoid thread race issues in detector session
                if frame_idx % detect_interval == 0:
                    try:
                        detected = detect_faces_in_frame(frame)
                        if detected:
                            cached_faces = _match_faces(detected, cached_faces)
                    except Exception as det_err:
                        logger.warning(f'Frame {frame_idx}: detection failed – {det_err}')

                # Submit frame swapping to the thread pool executor
                target_faces_to_swap = list(cached_faces) if cached_faces else []
                
                future = executor.submit(
                    swap_frame,
                    source_face, target_faces_to_swap, frame,
                    face_indices=face_indices, enhance=enhance,
                    gender_match_mode=gender_match_mode
                )
                futures[frame_idx] = future
                frame_idx += 1

                # Periodically write finished frames sequentially in order
                while write_idx in processed_frames or (write_idx in futures and futures[write_idx].done()):
                    if write_idx in processed_frames:
                        frame_to_write = processed_frames.pop(write_idx)
                    else:
                        try:
                            frame_to_write = futures[write_idx].result()
                        except Exception as fut_err:
                            logger.error(f"Failed to process frame {write_idx}: {fut_err}")
                            frame_to_write = frame # fallback to original resized frame
                            skip_count += 1
                        del futures[write_idx]
                    writer.write(frame_to_write)
                    write_idx += 1

                # Update progress + live ETA every 3 frames
                if job_id and write_idx % 3 == 0 and write_idx > 0:
                    elapsed = time.time() - t_loop_start
                    fps_proc = write_idx / elapsed if elapsed > 0 else 0
                    remaining = (total - write_idx) / fps_proc if fps_proc > 0 and total > write_idx else 0
                    _job_progress[job_id] = {
                        'total': total or write_idx,
                        'done': write_idx,
                        'status': 'processing',
                        'eta_seconds': int(remaining),
                        'fps_proc': round(fps_proc, 2),
                        'skipped': skip_count,
                    }

            # Retrieve any remaining completed tasks
            for fid, fut in list(futures.items()):
                try:
                    processed_frames[fid] = fut.result()
                except Exception as fut_err:
                    logger.error(f"Failed to process frame {fid}: {fut_err}")
                    skip_count += 1
                del futures[fid]

            # Write out all remaining frames sequentially
            while write_idx < frame_idx:
                if write_idx in processed_frames:
                    writer.write(processed_frames.pop(write_idx))
                write_idx += 1

        cap.release()
        writer.release()

        if job_id:
            _job_progress[job_id] = {'total': frame_idx, 'done': frame_idx,
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
            _job_progress[job_id] = {'total': frame_idx, 'done': frame_idx,
                                      'status': 'done', 'eta_seconds': 0, 'fps_proc': 0}

        elapsed_total = time.time() - t0
        return {
            'success': True,
            'processing_time': round(elapsed_total, 2),
            'frames_processed': frame_idx,
            'frames_skipped': skip_count,
            'fps_achieved': round(frame_idx / elapsed_total, 2) if elapsed_total > 0 else 0,
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
    Detect faces in an uploaded image or video (first frame).
    Returns list of face crops (base64 JPEG thumbnails) and the saved filename.
    """
    file_key = 'target' if 'target' in request.files else ('image' if 'image' in request.files else None)
    if not file_key:
        return jsonify({'success': False, 'error': 'No target file uploaded'}), 400

    tgt_file = request.files[file_key]
    if not allowed_file(tgt_file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400

    fname = secure_filename(f'detect_{int(time.time())}_{tgt_file.filename}')
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    tgt_file.save(path)

    frame = read_frame(path)
    if frame is None:
        return jsonify({'success': False, 'error': 'Cannot read target file'}), 400

    faces = detect_faces_in_frame(frame)
    crops = []
    for face in faces:
        crop = _crop_face(frame, face)
        crops.append({
            'b64': _frame_to_b64(crop),
            'score': float(face.score_set.get('detector', 0)),
            'bbox': [float(v) for v in face.bounding_box],
            'gender': getattr(face, 'gender', 'unknown')
        })

    return jsonify({
        'success': True,
        'count': len(faces),
        'faces': crops,
        'image_path': fname,
    })


@app.route('/api/detect-source-faces', methods=['POST'])
def api_detect_source_faces():
    """
    Detect faces in uploaded source images.
    Returns list of face crops (base64 JPEG thumbnails) and the saved filenames.
    """
    if 'sources' not in request.files:
        if 'source' in request.files:
            files = [request.files['source']]
        else:
            return jsonify({'success': False, 'error': 'No source files uploaded'}), 400
    else:
        files = request.files.getlist('sources')
        
    if not files or len(files) == 0:
        return jsonify({'success': False, 'error': 'No files uploaded'}), 400

    detected_faces = []
    ts = int(time.time())
    
    for idx, f in enumerate(files):
        if not allowed_file(f.filename):
            continue
        fname = secure_filename(f'source_{ts}_{idx}_{f.filename}')
        path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        f.save(path)
        
        frame = read_frame(path)
        if frame is None:
            continue
            
        faces = detect_faces_in_frame(frame)
        if not faces:
            continue

        face = pick_source_face(faces)
        crop = _crop_face(frame, face)
        detected_faces.append({
            'source_idx': idx,
            'filename': fname,
            'b64': _frame_to_b64(crop),
            'gender': getattr(face, 'gender', 'unknown'),
            'score': float(face.score_set.get('detector', 0))
        })
        
    return jsonify({
        'success': True,
        'faces': detected_faces
    })


def swap_frame_multiple(sources_faces_dict: dict, target_faces, frame: np.ndarray,
                        mapping: dict, enhance: bool = True,
                        gender_match_mode: str = 'features_only',
                        stats: dict | None = None) -> np.ndarray:
    """
    Apply mapped source faces onto selected target faces in frame.
    mapping: dict of { target_face_idx (int/str): source_filename (str) }
    sources_faces_dict: dict of { source_filename (str): Face object }
    `stats`: optional accounting dict (see swap_frame) so callers can detect a
             no-op swap instead of silently returning the untouched target.
    """
    result = frame.copy()

    # Save default masking states to prevent thread race contamination
    orig_mask_types = state_manager.get_item('face_mask_types') or ['box']
    orig_mask_regions = state_manager.get_item('face_mask_regions') or []
    orig_mask_blur = state_manager.get_item('face_mask_blur') or 0.3
    orig_mask_padding = state_manager.get_item('face_mask_padding') or [0, 0, 0, 0]
    orig_swapper_weight = state_manager.get_item('face_swapper_weight')
    orig_enhancer_weight = state_manager.get_item('face_enhancer_weight')

    for t_idx_str, src_fname in mapping.items():
        try:
            t_idx = int(t_idx_str)
        except ValueError:
            continue

        if t_idx < 0 or t_idx >= len(target_faces):
            continue

        tface = target_faces[t_idx]
        source_face = sources_faces_dict.get(src_fname)
        if source_face is None:
            continue

        if stats is not None:
            stats['attempted'] = stats.get('attempted', 0) + 1

        source_gender = getattr(source_face, 'gender', None)
        target_gender = getattr(tface, 'gender', None)
        is_cross_gender = bool(source_gender and target_gender and source_gender != target_gender)

        if gender_match_mode == 'match' and is_cross_gender:
            logger.info(f'Skipping swap for target face {t_idx} due to gender mismatch ({source_gender} -> {target_gender})')
            if stats is not None:
                stats['skipped_gender'] = stats.get('skipped_gender', 0) + 1
            continue

        try:
            # Full-face box swap at full source identity for every face (see swap_frame).
            state_manager.set_item('face_mask_types', ['box'])
            state_manager.set_item('face_mask_blur', 0.3)
            state_manager.set_item('face_mask_padding', [0, 0, 0, 0])
            state_manager.set_item('face_swapper_weight', orig_swapper_weight)
            state_manager.set_item('face_enhancer_weight', orig_enhancer_weight)

            result = face_swapper.swap_face(source_face, tface, result)
            if enhance and _ENHANCER_ACTIVE:
                try:
                    result = face_enhancer.enhance_face(tface, result)
                except Exception as ee:
                    logger.warning(f'enhance failed for target {t_idx} (kept swap): {ee}')
            if stats is not None:
                stats['swapped'] = stats.get('swapped', 0) + 1
        except Exception as e:
            logger.warning(f'swap_frame_multiple: swap/enhance failed for target face {t_idx}: {e}\n{traceback.format_exc()}')
            if stats is not None:
                stats.setdefault('errors', []).append(f'target {t_idx}: {e}')

    # Restore original states
    state_manager.set_item('face_mask_types', orig_mask_types)
    state_manager.set_item('face_mask_regions', orig_mask_regions)
    state_manager.set_item('face_mask_blur', orig_mask_blur)
    state_manager.set_item('face_mask_padding', orig_mask_padding)
    state_manager.set_item('face_swapper_weight', orig_swapper_weight)
    state_manager.set_item('face_enhancer_weight', orig_enhancer_weight)

    return result


def process_multiple_swaps(sources_faces_dict: dict, target_path: str, output_path: str,
                           mapping: dict, enhance: bool = True,
                           pitch_semitones: float = 0.0,
                           job_id: str = '',
                           detect_interval: int = 5,
                           max_side: int = 720,
                           gender_match_mode: str = 'features_only') -> dict:
    """
    Image/video processing for multiple source to target face mappings.
    """
    t0 = time.time()

    # If using features-only mode, ensure Bisenet model is downloaded/present
    if gender_match_mode in ('features_only', 'always_features_only'):
        state_manager.set_item('face_mask_types', ['region'])
        state_manager.set_item('face_mask_regions', CROSS_GENDER_REGIONS)
        if not check_bisenet_model():
            return {'success': False, 'error': 'Failed to download or initialize face parser models (Bisenet).'}

    # ── Image swap ───────────────────────────────────────────────────────────
    if is_image(target_path):
        state_manager.set_item('face_swapper_pixel_boost', _image_pixel_boost('512x512'))
        target_frame = read_frame(target_path)
        if target_frame is None:
            return {'success': False, 'error': 'Cannot read target image'}

        target_faces = detect_faces_in_frame(target_frame)
        if not target_faces:
            return {'success': False, 'error': 'No face detected in the target image'}

        stats: dict = {}
        result = swap_frame_multiple(sources_faces_dict, target_faces, target_frame,
                                     mapping=mapping, enhance=enhance,
                                     gender_match_mode=gender_match_mode, stats=stats)

        # Honest failure instead of silently returning the untouched target.
        if stats.get('swapped', 0) == 0:
            if stats.get('skipped_gender', 0) > 0:
                return {'success': False, 'error':
                        'All mapped faces were skipped due to gender mismatch. '
                        'Switch to "realistic" or "faithful" mode to swap across genders.'}
            errs = '; '.join(stats.get('errors', [])) or 'unknown error'
            return {'success': False, 'error': f'Face swap failed — no faces were swapped ({errs}).'}

        # Apply LAB colour transfer for all swapped target faces
        active_faces = [target_faces[int(t_idx)] for t_idx in mapping.keys()
                        if 0 <= int(t_idx) < len(target_faces)]
        result = _advanced_face_blend(result, target_frame, active_faces)

        ok = write_image(output_path, result)
        if not ok:
            return {'success': False, 'error': 'Failed to write output image'}

        return {
            'success': True,
            'processing_time': round(time.time() - t0, 2),
            'faces_swapped': stats.get('swapped', 0),
            'output_type': 'image'
        }

    # ── Video swap ───────────────────────────────────────────────────────────
    if is_video(target_path):
        cap = cv2.VideoCapture(target_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_side > 0 and max(src_w, src_h) > max_side:
            scale = max_side / max(src_w, src_h)
            out_w, out_h = int(src_w * scale) & ~1, int(src_h * scale) & ~1
        else:
            out_w, out_h = src_w & ~1, src_h & ~1
            
        logger.info(f'Video Multi-Swap: {src_w}×{src_h} → {out_w}×{out_h}, {total} frames @ {fps:.1f}fps')
        state_manager.set_item('face_swapper_pixel_boost', _image_pixel_boost('128x128'))

        base, _ = os.path.splitext(output_path)
        tmp_video = base + '_raw.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(tmp_video, fourcc, fps, (out_w, out_h))

        # Validate first frame
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

        max_workers = max(1, int(state_manager.get_item('execution_thread_count') or 4))
        
        futures = {}
        processed_frames = {}
        frame_idx = 0
        write_idx = 0
        
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while True:
                while len(futures) >= max_workers * 2:
                    time.sleep(0.005)
                    completed_ids = [fid for fid, fut in futures.items() if fut.done()]
                    for fid in completed_ids:
                        try:
                            processed_frames[fid] = futures[fid].result()
                        except Exception as fut_err:
                            logger.error(f"Failed to process frame {fid}: {fut_err}")
                            skip_count += 1
                        del futures[fid]

                ok, frame = cap.read()
                if not ok:
                    break
                frame = _resize_frame(frame, max_side)

                if frame_idx % detect_interval == 0:
                    try:
                        detected = detect_faces_in_frame(frame)
                        if detected:
                            cached_faces = _match_faces(detected, cached_faces)
                    except Exception as det_err:
                        logger.warning(f'Frame {frame_idx}: detection failed – {det_err}')

                target_faces_to_swap = list(cached_faces) if cached_faces else []
                
                future = executor.submit(
                    swap_frame_multiple,
                    sources_faces_dict, target_faces_to_swap, frame,
                    mapping=mapping, enhance=enhance,
                    gender_match_mode=gender_match_mode
                )
                futures[frame_idx] = future
                frame_idx += 1

                while write_idx in processed_frames or (write_idx in futures and futures[write_idx].done()):
                    if write_idx in processed_frames:
                        frame_to_write = processed_frames.pop(write_idx)
                    else:
                        try:
                            frame_to_write = futures[write_idx].result()
                        except Exception as fut_err:
                            logger.error(f"Failed to process frame {write_idx}: {fut_err}")
                            frame_to_write = frame
                            skip_count += 1
                        del futures[write_idx]
                    writer.write(frame_to_write)
                    write_idx += 1

                if job_id and write_idx % 3 == 0 and write_idx > 0:
                    elapsed = time.time() - t_loop_start
                    fps_proc = write_idx / elapsed if elapsed > 0 else 0
                    remaining = (total - write_idx) / fps_proc if fps_proc > 0 and total > write_idx else 0
                    _job_progress[job_id] = {
                        'total': total or write_idx,
                        'done': write_idx,
                        'status': 'processing',
                        'eta_seconds': int(remaining),
                        'fps_proc': round(fps_proc, 2),
                        'skipped': skip_count,
                    }

            for fid, fut in list(futures.items()):
                try:
                    processed_frames[fid] = fut.result()
                except Exception as fut_err:
                    logger.error(f"Failed to process frame {fid}: {fut_err}")
                    skip_count += 1
                del futures[fid]

            while write_idx < frame_idx:
                if write_idx in processed_frames:
                    writer.write(processed_frames.pop(write_idx))
                write_idx += 1

        cap.release()
        writer.release()

        if job_id:
            _job_progress[job_id] = {'total': frame_idx, 'done': frame_idx,
                                      'status': 'encoding', 'eta_seconds': 0, 'fps_proc': 0}

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
            _job_progress[job_id] = {'total': frame_idx, 'done': frame_idx,
                                      'status': 'done', 'eta_seconds': 0, 'fps_proc': 0}

        elapsed_total = time.time() - t0
        return {
            'success': True,
            'processing_time': round(elapsed_total, 2),
            'frames_processed': frame_idx,
            'frames_skipped': skip_count,
            'fps_achieved': round(frame_idx / elapsed_total, 2) if elapsed_total > 0 else 0,
            'faces_swapped': len(mapping),
            'output_type': 'video',
            'output_resolution': f'{out_w}x{out_h}',
        }

    return {'success': False, 'error': 'Unsupported target file type'}


@app.route('/api/swap-multiple', methods=['POST'])
def api_swap_multiple():
    """
    Perform face swap with multiple target-to-source mappings.
    """
    target_filename = request.form.get('target_filename', '')
    mapping_raw = request.form.get('mapping', '')
    
    if not target_filename or not mapping_raw:
        return jsonify({'success': False, 'error': 'Missing target filename or mapping'}), 400
        
    try:
        import json
        mapping = json.loads(mapping_raw)
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid mapping format'}), 400
        
    if not isinstance(mapping, dict):
        return jsonify({'success': False, 'error': 'Mapping must be a JSON dictionary'}), 400

    target_path = os.path.join(app.config['UPLOAD_FOLDER'], target_filename)
    if not os.path.isfile(target_path):
        return jsonify({'success': False, 'error': 'Target file not found on server'}), 400

    unique_sources = set(mapping.values())
    sources_faces_dict = {}
    
    for src_fname in unique_sources:
        if not src_fname:
            continue
        src_path = os.path.join(app.config['UPLOAD_FOLDER'], src_fname)
        if not os.path.isfile(src_path):
            return jsonify({'success': False, 'error': f'Source file {src_fname} not found on server'}), 400
            
        frame = read_frame(src_path)
        if frame is None:
            return jsonify({'success': False, 'error': f'Cannot read source file {src_fname}'}), 400
            
        faces = detect_faces_in_frame(frame)
        if not faces:
            return jsonify({'success': False, 'error': f'No face detected in source {src_fname}'}), 400
            
        sources_faces_dict[src_fname] = faces[0]

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

    ts = int(time.time())
    tgt_ext = os.path.splitext(target_filename)[1].lower()
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
    
    gender_match_mode = _resolve_swap_mode(request.form)

    result = process_multiple_swaps(sources_faces_dict, target_path, out_path,
                                    mapping=mapping,
                                    enhance=enhance,
                                    pitch_semitones=pitch,
                                    job_id=job_id,
                                    detect_interval=detect_interval,
                                    max_side=max_side,
                                    gender_match_mode=gender_match_mode)
                                    
    if result['success']:
        result['output_filename'] = out_name
        result['output_url'] = f'/output/{out_name}'
        if result.get('output_type') == 'image' and os.path.isfile(out_path):
            with open(out_path, 'rb') as f:
                result['preview_b64'] = base64.b64encode(f.read()).decode()
                
    return jsonify(result), 200 if result['success'] else 500


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

    gender_match_mode = _resolve_swap_mode(request.form)

    # If using features-only mode, ensure Bisenet model is downloaded/present
    if gender_match_mode in ('features_only', 'always_features_only'):
        state_manager.set_item('face_mask_types', ['region'])
        state_manager.set_item('face_mask_regions', CROSS_GENDER_REGIONS)
        if not check_bisenet_model():
            return jsonify({'success': False, 'error': 'Failed to download or initialize face parser models (Bisenet) required for features-only swap.'}), 500

    result = process_image_swap(src_path, tgt_path, out_path,
                                face_indices=face_indices,
                                enhance=enhance,
                                pitch_semitones=pitch,
                                job_id=job_id,
                                detect_interval=detect_interval,
                                max_side=max_side,
                                gender_match_mode=gender_match_mode)
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

    src_face = pick_source_face(faces)
    state_manager.set_item('source_paths', [path])
    with _webcam_lock:
        _webcam_state['source_face'] = src_face
        _webcam_state['source_path'] = path
        _webcam_state['last_result'] = None

    crop = _crop_face(frame, src_face)
    return jsonify({
        'success': True,
        'preview_b64': _frame_to_b64(crop),
        'score': float(src_face.score_set.get('detector', 0)),
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
        <label>Gender Handling (automatic)</label>
        <select id="genderMatchMode" style="width:100%;background:var(--surface2);color:var(--text);border:1px solid #ffffff20;border-radius:6px;padding:8px;font-size:.9rem">
          <option value="realistic" selected>Automatic — detect gender & prevent beard (recommended)</option>
          <option value="faithful">Faithful — full identity transfer (keeps source beard)</option>
          <option value="match">Match genders only (skip cross-gender faces)</option>
        </select>
        <div style="font-size:.75rem;color:var(--muted);margin-top:4px">Leave on <b>Automatic</b> — the app detects each face's gender and only suppresses beard/lips on cross-gender swaps, so a male beard never lands on a female face (or female lips on a male). No need to choose. Pick Faithful only if you want the full face incl. facial hair.</div>
      </div>
    </div>
    
    <div class="grid2" style="margin-top:14px">
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
      <h3>Source Faces (Select one or more)</h3>
      <div class="upload-zone" onclick="document.getElementById('mfSrcFile').click()">
        <div class="icon">🤳</div><p>Click or drag portrait photos</p>
        <input type="file" id="mfSrcFile" accept="image/*" multiple>
      </div>
      <div id="mfSourceGrid" class="face-grid" style="margin-top:14px">
        <span style="color:var(--muted);font-size:.85rem">Upload source portraits…</span>
      </div>
    </div>

    <div class="card">
      <h3>Target Image or Video (multi-face)</h3>
      <div class="upload-zone" id="mfTgtZone" onclick="document.getElementById('mfTgtFile').click()">
        <div class="icon">👥</div><p>Click or drag target image or video</p>
        <input type="file" id="mfTgtFile" accept="image/*,video/*">
      </div>
      <img id="mfTgtPreview" class="preview-img">
    </div>
  </div>

  <!-- Multi-face Options -->
  <div class="card" style="margin-top:16px">
    <h3>Multi-Face Swap Options</h3>
    <div class="grid2">
      <div>
        <label>Face Enhancer</label>
        <label style="display:flex;align-items:center;gap:8px;margin-top:4px">
          <input type="checkbox" id="mfEnhanceCheck" checked style="accent-color:var(--accent)">
          <span style="font-size:.9rem">Enable GFPGAN (slower, higher quality)</span>
        </label>
        <div style="font-size:.75rem;color:var(--muted);margin-top:4px" id="mfEnhanceHint">ON by default for images — auto-disabled for video</div>
      </div>
      <div>
        <label>Gender Handling (automatic)</label>
        <select id="mfGenderMatchMode" style="width:100%;background:var(--surface2);color:var(--text);border:1px solid #ffffff20;border-radius:6px;padding:8px;font-size:.9rem">
          <option value="realistic" selected>Automatic — detect gender & prevent beard (recommended)</option>
          <option value="faithful">Faithful — full identity transfer (keeps source beard)</option>
          <option value="match">Match genders only (skip cross-gender faces)</option>
        </select>
        <div style="font-size:.75rem;color:var(--muted);margin-top:4px">Leave on <b>Automatic</b> — the app detects each face's gender and only suppresses beard/lips on cross-gender swaps, so a male beard never lands on a female face (or female lips on a male). No need to choose. Pick Faithful only if you want the full face incl. facial hair.</div>
      </div>
    </div>
    
    <div class="grid2" style="margin-top:14px">
      <div>
        <label>Audio Pitch Shift (video only)</label>
        <div class="range-row">
          <input type="range" id="mfPitchRange" min="-12" max="12" step="1" value="0"
                 oninput="document.getElementById('mfPitchVal').textContent=this.value">
          <span class="range-val" id="mfPitchVal">0</span>
          <span style="font-size:.8rem;color:var(--muted)">semitones</span>
        </div>
      </div>
    </div>

    <div id="mfVideoOpts" style="display:none;margin-top:16px;padding-top:14px;border-top:1px solid #ffffff12">
      <div style="font-size:.8rem;color:var(--accent);font-weight:700;margin-bottom:10px">⚡ Video Speed Settings</div>
      <div class="grid2">
        <div>
          <label>Face Detect Interval</label>
          <select id="mfDetectInterval" style="width:100%;background:var(--surface2);color:var(--text);border:1px solid #ffffff20;border-radius:6px;padding:8px;font-size:.9rem">
            <option value="1">Every frame (slowest, most accurate)</option>
            <option value="3" selected>Every 3 frames (recommended)</option>
            <option value="5">Every 5 frames (faster)</option>
            <option value="10">Every 10 frames (fastest)</option>
            <option value="30">Every 30 frames (ultra fast)</option>
          </select>
        </div>
        <div>
          <label>Max Resolution</label>
          <select id="mfMaxSide" style="width:100%;background:var(--surface2);color:var(--text);border:1px solid #ffffff20;border-radius:6px;padding:8px;font-size:.9rem">
            <option value="480">480p (fastest)</option>
            <option value="720" selected>720p (recommended)</option>
            <option value="1080">1080p</option>
            <option value="0">Original (no resize)</option>
          </select>
        </div>
      </div>
      <div id="mfTimeEstimate" style="margin-top:12px;padding:10px 14px;background:var(--surface2);border-radius:8px;font-size:.85rem;color:var(--muted)">
        Select a video to see estimated processing time
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h3>Detected Target Faces – Map each to a Source Face</h3>
    <div id="mfFaceGrid" class="face-grid" style="gap:24px 16px">
      <span style="color:var(--muted);font-size:.85rem">Upload a target image or video to detect faces…</span>
    </div>
    <div style="margin-top:12px;font-size:.82rem;color:var(--muted)" id="mfSelInfo">
      0 target faces mapped
    </div>
  </div>

  <div style="margin-top:20px;display:flex;gap:12px;flex-wrap:wrap">
    <button class="btn btn-primary" onclick="doMultiSwap()" id="mfSwapBtn">⚡ Swap Selected Faces</button>
  </div>

  <div id="mfAlert" class="alert"></div>
  <div class="progress" id="mfProgress"><div class="progress-bar" id="mfBar" style="width:0%"></div></div>

  <div class="output-box" id="mfOutput">
    <h3 style="font-size:.85rem;color:var(--muted);margin-bottom:10px">RESULT</h3>
    <img id="mfOutImg" style="display:none;max-width:100%;border-radius:10px;box-shadow:0 4px 20px rgba(0,0,0,.4)">
    <video id="mfOutVid" controls style="display:none;max-width:100%;border-radius:10px"></video>
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
  (function(m){ if(m==='realistic'||m==='faithful') fd.append('realism_mode', m); else fd.append('gender_match_mode', m); })(document.getElementById('genderMatchMode').value);
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
let mfSourceFaces = [];
let mfTargetFaces = [];
let mfTgtFilename = null;

document.getElementById('mfSrcFile').addEventListener('change', async function(){
  const files = this.files;
  if(!files.length) return;
  
  const fd = new FormData();
  for(let i=0; i<files.length; i++){
    fd.append('sources', files[i]);
  }
  
  const grid = document.getElementById('mfSourceGrid');
  grid.innerHTML = '<span class="spinner"></span> Detecting source faces…';
  showAlert('mfAlert', 'Detecting faces in source images…', 'info');
  
  try {
    const res = await fetch('/api/detect-source-faces', {method: 'POST', body: fd});
    const data = await res.json();
    if(data.success && data.faces.length > 0){
      hideAlert('mfAlert');
      mfSourceFaces = data.faces.map((f, idx) => ({
        ...f,
        label: `S${idx + 1}`
      }));
      renderSourceGrid();
      if(mfTargetFaces.length > 0) {
        renderTargetMappingGrid();
      }
    } else {
      showAlert('mfAlert', '❌ ' + (data.error || 'No faces found in sources'), 'error');
      grid.innerHTML = '<span style="color:var(--muted);font-size:.85rem">No faces found</span>';
    }
  } catch(e) {
    showAlert('mfAlert', 'Error: ' + e.message, 'error');
    grid.innerHTML = '<span style="color:var(--muted);font-size:.85rem">Error detecting faces</span>';
  }
});

function renderSourceGrid(){
  const grid = document.getElementById('mfSourceGrid');
  grid.innerHTML = '';
  mfSourceFaces.forEach((f) => {
    const container = document.createElement('div');
    container.style.cssText = 'position:relative;display:inline-block;';
    
    const img = document.createElement('img');
    img.src = 'data:image/jpeg;base64,' + f.b64;
    img.className = 'face-thumb';
    img.style.borderColor = 'var(--success)';
    img.title = `Source Face ${f.label} (${f.gender})`;
    
    const label = document.createElement('span');
    label.textContent = f.label;
    label.style.cssText = 'position:absolute;bottom:4px;right:4px;background:var(--success);color:#fff;font-size:10px;font-weight:bold;padding:2px 6px;border-radius:4px;';
    
    container.appendChild(img);
    container.appendChild(label);
    grid.appendChild(container);
  });
}

document.getElementById('mfTgtFile').addEventListener('change', async function(){
  const file = this.files[0];
  if(!file) return;
  
  const url = URL.createObjectURL(file);
  const isVideo = /\.(mp4|avi|mov|mkv)$/i.test(file.name) || file.type.startsWith('video/');
  
  const prevZone = document.getElementById('mfTgtZone');
  prevZone.querySelectorAll('img.preview-img, video.preview-img').forEach(el=>el.remove());
  
  if(isVideo){
    document.getElementById('mfEnhanceCheck').checked = false;
    document.getElementById('mfEnhanceHint').textContent = 'OFF by default for video — enable only for short clips';
    const v = document.createElement('video');
    v.src = url; v.controls = true; v.muted = true; v.className = 'preview-img';
    v.style.cssText = 'display:block;max-width:100%;max-height:220px;border-radius:8px;margin-top:10px';
    v.addEventListener('loadedmetadata', ()=>{
      const estFrames = Math.round(v.duration * 30);
      document.getElementById('mfVideoOpts').style.display = 'block';
      updateMfTimeEstimate(estFrames);
    });
    prevZone.appendChild(v);
  } else {
    document.getElementById('mfEnhanceCheck').checked = true;
    document.getElementById('mfEnhanceHint').textContent = 'ON by default for images — auto-disabled for video';
    document.getElementById('mfVideoOpts').style.display = 'none';
    const img = document.getElementById('mfTgtPreview');
    img.src = url; img.style.display='block';
  }
  
  document.getElementById('mfOutput').style.display = 'none';
  hideAlert('mfAlert');
  
  const fd = new FormData();
  fd.append('target', file);
  
  const grid = document.getElementById('mfFaceGrid');
  grid.innerHTML = '<span class="spinner"></span> Detecting target faces…';
  showAlert('mfAlert', 'Detecting faces in target media…', 'info');
  
  try {
    const res = await fetch('/api/detect-faces', {method: 'POST', body: fd});
    const data = await res.json();
    if(data.success && data.faces.length > 0){
      hideAlert('mfAlert');
      mfTgtFilename = data.image_path;
      mfTargetFaces = data.faces;
      renderTargetMappingGrid();
    } else {
      showAlert('mfAlert', '❌ ' + (data.error || 'No faces found in target'), 'error');
      grid.innerHTML = '<span style="color:var(--muted);font-size:.85rem">No target faces found</span>';
    }
  } catch(e) {
    showAlert('mfAlert', 'Error: ' + e.message, 'error');
    grid.innerHTML = '<span style="color:var(--muted);font-size:.85rem">Error detecting faces</span>';
  }
});

function renderTargetMappingGrid(){
  const grid = document.getElementById('mfFaceGrid');
  if(!mfTargetFaces.length){
    grid.innerHTML = '<span style="color:var(--muted);font-size:.85rem">No target faces detected</span>';
    return;
  }
  
  grid.innerHTML = '';
  mfTargetFaces.forEach((f, idx) => {
    const block = document.createElement('div');
    block.style.cssText = 'display:flex;flex-direction:column;align-items:center;background:var(--surface2);padding:12px;border-radius:8px;border:1px solid #ffffff10;width:120px;';
    
    const img = document.createElement('img');
    img.src = 'data:image/jpeg;base64,' + f.b64;
    img.style.cssText = 'width:80px;height:80px;border-radius:8px;object-fit:cover;margin-bottom:8px;';
    
    const info = document.createElement('div');
    info.textContent = `T${idx+1} (${f.gender || '?'})`;
    info.style.cssText = 'font-size:11px;color:var(--muted);margin-bottom:6px;text-align:center;width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
    
    const select = document.createElement('select');
    select.className = 'target-mapping-select';
    select.dataset.targetIdx = idx;
    select.style.cssText = 'width:100%;background:var(--surface);color:var(--text);border:1px solid #ffffff15;border-radius:4px;padding:4px;font-size:11px;outline:none;cursor:pointer;';
    
    const optDefault = document.createElement('option');
    optDefault.value = '';
    optDefault.textContent = "Don't swap";
    select.appendChild(optDefault);
    
    mfSourceFaces.forEach(sf => {
      const opt = document.createElement('option');
      opt.value = sf.filename;
      opt.textContent = `Swap ${sf.label} (${sf.gender || '?'})`;
      select.appendChild(opt);
    });
    
    if(mfSourceFaces.length > 0) {
      const matchedSources = mfSourceFaces.filter(sf => sf.gender === f.gender);
      if(matchedSources.length === 1) {
        select.value = matchedSources[0].filename;
      }
    }
    
    select.addEventListener('change', updateMfSelInfo);
    
    block.appendChild(img);
    block.appendChild(info);
    block.appendChild(select);
    grid.appendChild(block);
  });
  
  updateMfSelInfo();
}

function updateMfSelInfo(){
  const selects = document.querySelectorAll('.target-mapping-select');
  let mappedCount = 0;
  selects.forEach(sel => {
    if(sel.value) mappedCount++;
  });
  
  document.getElementById('mfSelInfo').textContent = `${mappedCount} of ${mfTargetFaces.length} target faces mapped`;
}

function updateMfTimeEstimate(frames){
  const enhance = document.getElementById('mfEnhanceCheck').checked;
  const interval = parseInt(document.getElementById('mfDetectInterval').value || '3');
  const maxSide  = parseInt(document.getElementById('mfMaxSide').value || '720');

  const detectCost = 0.25;
  const swapCost   = 0.65;
  const enhCost    = 3.0;
  const resizeFactor = maxSide > 0 ? Math.min(1.0, (maxSide / 1080) ** 1.5) : 1.0;

  const perFrame = (detectCost / interval + swapCost + (enhance ? enhCost : 0)) * resizeFactor;
  const totalSec = Math.round(frames * perFrame);

  let timeStr;
  if(totalSec < 60) timeStr = `~${totalSec}s`;
  else if(totalSec < 3600) timeStr = `~${Math.round(totalSec/60)} min`;
  else timeStr = `~${(totalSec/3600).toFixed(1)} hr`;

  const fps = (1/perFrame).toFixed(2);
  document.getElementById('mfTimeEstimate').innerHTML =
    `⏱ Estimated: <strong style="color:var(--accent)">${timeStr}</strong> &nbsp;|&nbsp; `+
    `${frames} frames &nbsp;|&nbsp; ~${fps} frames/sec &nbsp;|&nbsp; `+
    `<span style="color:var(--warn)">Tip: disable enhancer + detect every 10 frames for max speed</span>`;
}

// Update live when settings change
['mfDetectInterval','mfMaxSide','mfEnhanceCheck'].forEach(id=>{
  const el = document.getElementById(id);
  if(el) el.addEventListener('change', ()=>{
    const v = document.querySelector('#mfTgtZone video.preview-img');
    if(v && v.duration) updateMfTimeEstimate(Math.round(v.duration * 30));
  });
});

async function doMultiSwap(){
  if(!mfTgtFilename){
    showAlert('mfAlert', 'Please upload a target image or video first', 'warn');
    return;
  }
  
  const mapping = {};
  const selects = document.querySelectorAll('.target-mapping-select');
  selects.forEach(sel => {
    if(sel.value) {
      mapping[sel.dataset.targetIdx] = sel.value;
    }
  });
  
  if(Object.keys(mapping).length === 0){
    showAlert('mfAlert', 'Please map at least one target face to a source face', 'warn');
    return;
  }
  
  const isVideo = document.querySelector('#mfTgtZone video.preview-img') !== null;
  const jobId = Math.random().toString(36).substr(2,9);

  const btn = document.getElementById('mfSwapBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Uploading…';
  hideAlert('mfAlert');
  document.getElementById('mfOutput').style.display='none';

  const prog = document.getElementById('mfProgress');
  const bar  = document.getElementById('mfBar');
  prog.style.display = 'block';
  bar.style.width = '0%';
  
  if(isVideo){
    showAlert('mfAlert',
      '⏳ Video detected. Processing can take several minutes on GPU. Each frame is processed individually.',
      'info');
  }

  const fd = new FormData();
  fd.append('target_filename', mfTgtFilename);
  fd.append('mapping', JSON.stringify(mapping));
  fd.append('enhance', document.getElementById('mfEnhanceCheck').checked ? '1' : '0');
  fd.append('pitch_semitones', document.getElementById('mfPitchRange').value);
  (function(m){ if(m==='realistic'||m==='faithful') fd.append('realism_mode', m); else fd.append('gender_match_mode', m); })(document.getElementById('mfGenderMatchMode').value);
  fd.append('job_id', jobId);

  if(isVideo){
    fd.append('detect_interval', document.getElementById('mfDetectInterval').value || '5');
    fd.append('max_side', document.getElementById('mfMaxSide').value || '720');
  }

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

  if(isVideo) localStorage.setItem('deeptrace_job_id', jobId);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/swap-multiple');
  xhr.timeout = 3600000; // 1 hour

  xhr.upload.onprogress = (e)=>{
    if(!e.lengthComputable) return;
    const pct = Math.round((e.loaded/e.total) * (isVideo ? 15 : 80));
    bar.style.width = pct+'%';
  };

  xhr.upload.onload = ()=>{
    if(isVideo){
      bar.style.width = '20%';
      btn.innerHTML = '<span class="spinner"></span> Processing frames…';
    } else {
      bar.style.width = '50%';
      btn.innerHTML = '<span class="spinner"></span> Swapping faces…';
    }
  };

  xhr.onload = ()=>{
    if(pollTimer) clearInterval(pollTimer);
    localStorage.removeItem('deeptrace_job_id');
    bar.style.width = '100%';
    setTimeout(()=>{ prog.style.display='none'; bar.style.width='0%'; }, 800);
    btn.disabled=false; btn.innerHTML='⚡ Swap Selected Faces';
    
    try{
      const data = JSON.parse(xhr.responseText);
      if(data.success){
        hideAlert('mfAlert');
        const extra = isVideo
          ? ` | ${data.frames_processed} frames @ ${data.fps_achieved} fps (${data.output_resolution||''})`
          : '';
        showAlert('mfAlert',`✅ Done in ${data.processing_time}s — ${data.faces_swapped} face(s) swapped${extra}`,'success');
        
        const box = document.getElementById('mfOutput');
        box.style.display = 'block';
        const img = document.getElementById('mfOutImg');
        const vid = document.getElementById('mfOutVid');
        const dl  = document.getElementById('mfOutDl');
        const url = '/output/' + data.output_filename;
        
        if(isVideo){
          img.style.display = 'none';
          vid.style.display = 'block';
          vid.src = url;
          vid.load();
        } else {
          vid.style.display = 'none';
          img.style.display = 'block';
          if(data.preview_b64){
            const mime = data.output_filename.toLowerCase().endsWith('.png') ? 'image/png' : 
                         (data.output_filename.toLowerCase().endsWith('.webp') ? 'image/webp' : 'image/jpeg');
            img.src = `data:${mime};base64,${data.preview_b64}`;
          } else {
            img.src = url + '?t=' + Date.now();
          }
        }
        
        dl.href = url;
        dl.download = data.output_filename;
        
        const mfDlHtml = document.getElementById('mfOutDlHtml');
        if(mfDlHtml) {
          if(!isVideo && data.preview_b64) {
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
        
        document.getElementById('mfOutMeta').innerHTML =
          `<span class="stat">⏱ ${data.processing_time}s</span>
           <span class="stat">👤 ${data.faces_swapped} face(s)</span>`;
      } else {
        showAlert('mfAlert','❌ '+(data.error||'Swap failed'),'error');
      }
    } catch(e){ showAlert('mfAlert','Server error – check logs','error'); }
  };

  xhr.onerror = xhr.ontimeout = ()=>{
    if(pollTimer) clearInterval(pollTimer);
    prog.style.display='none';
    btn.disabled=false; btn.innerHTML='⚡ Swap Selected Faces';
    localStorage.removeItem('deeptrace_job_id');
    showAlert('mfAlert','Network error or timeout. Check server logs.','error');
  };

  xhr.send(fd);
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
