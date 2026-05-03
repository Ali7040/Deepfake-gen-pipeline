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
        state_manager.init_item('face_swapper_pixel_boost', '128x128')
        state_manager.init_item('face_swapper_weight', 0.5)

        # Face enhancer
        state_manager.init_item('face_enhancer_model', 'gfpgan_1.4')
        state_manager.init_item('face_enhancer_blend', 80)
        state_manager.init_item('face_enhancer_weight', 1.0)

        # Recogniser / landmarker
        state_manager.init_item('face_recognizer_model', 'arcface_inswapper')
        state_manager.init_item('face_landmarker_model', '2dfan4')
        state_manager.init_item('face_landmarker_score', 0.3)   # lowered from 0.5

        # Masks
        state_manager.init_item('face_mask_types', ['box'])
        state_manager.init_item('face_mask_blur', 0.3)
        state_manager.init_item('face_mask_padding', [0, 0, 0, 0])
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


def _shift_audio_pitch(src_video: str, dst_video: str, semitones: float) -> bool:
    """Pitch-shift the audio in `src_video` by `semitones` and write to `dst_video`."""
    if not _ffmpeg_available():
        return False
    rate = 2 ** (semitones / 12)
    # atempo can only go 0.5–2.0; chain two filters for larger shifts
    if 0.5 <= rate <= 2.0:
        atempo = f'atempo={rate:.4f}'
    elif rate < 0.5:
        atempo = f'atempo={rate**0.5:.4f},atempo={rate**0.5:.4f}'
    else:
        atempo = f'atempo={rate**0.5:.4f},atempo={rate**0.5:.4f}'

    cmd = [
        'ffmpeg', '-y',
        '-i', src_video,
        '-filter:a', f'asetrate=44100*{rate:.4f},{atempo},aresample=44100',
        '-c:v', 'copy',
        dst_video
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0
    except Exception as e:
        logger.error(f'Audio pitch shift failed: {e}')
        return False


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


def process_image_swap(source_path: str, target_path: str, output_path: str,
                       face_indices=None, enhance: bool = True,
                       pitch_semitones: float = 0.0) -> dict:
    """
    Full pipeline for image→image or video→video face swap.
    Returns a dict with keys: success, error, stats.
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
        target_frame = read_frame(target_path)
        if target_frame is None:
            return {'success': False, 'error': 'Cannot read target image'}

        target_faces = detect_faces_in_frame(target_frame)
        if not target_faces:
            return {'success': False, 'error': 'No face detected in the target image'}

        logger.info(f'{len(target_faces)} target face(s) detected')
        result = swap_frame(source_face, target_faces, target_frame,
                            face_indices=face_indices, enhance=enhance)
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
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        tmp_video = output_path.replace('.mp4', '_noaudio.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

        # Detect target faces from first frame to build index
        ok_first, first_frame = cap.read()
        if not ok_first:
            cap.release()
            return {'success': False, 'error': 'Cannot read first video frame'}
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        target_faces_first = detect_faces_in_frame(first_frame)
        if not target_faces_first:
            cap.release()
            return {'success': False, 'error': 'No face detected in target video'}

        frame_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Detect per frame so tracking remains accurate
            tfaces = detect_faces_in_frame(frame)
            if tfaces:
                frame = swap_frame(source_face, tfaces, frame,
                                   face_indices=face_indices, enhance=enhance)
            writer.write(frame)
            frame_count += 1
        cap.release()
        writer.release()

        # Audio handling
        final_output = output_path
        if pitch_semitones != 0.0 and _ffmpeg_available():
            ok = _shift_audio_pitch(target_path, output_path, pitch_semitones)
            if not ok:
                # fallback: copy video without audio shift
                subprocess.run(['ffmpeg', '-y', '-i', tmp_video,
                                '-c', 'copy', output_path],
                               capture_output=True, timeout=60)
        else:
            # Mux original audio back
            if _ffmpeg_available():
                cmd = ['ffmpeg', '-y',
                       '-i', tmp_video, '-i', target_path,
                       '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0?',
                       '-shortest', output_path]
                r = subprocess.run(cmd, capture_output=True, timeout=300)
                if r.returncode != 0:
                    import shutil
                    shutil.move(tmp_video, output_path)
            else:
                import shutil
                shutil.move(tmp_video, output_path)

        # Clean up temp file
        if Path(tmp_video).exists() and tmp_video != output_path:
            try:
                Path(tmp_video).unlink()
            except Exception:
                pass

        return {
            'success': True,
            'processing_time': round(time.time() - t0, 2),
            'frames_processed': frame_count,
            'faces_swapped': len([i for i in range(len(target_faces_first))
                                  if face_indices is None or i in face_indices]),
            'output_type': 'video'
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

    # Preserve original image format instead of forcing png
    tgt_ext = os.path.splitext(tgt_file.filename)[1].lower()
    valid_image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    valid_video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    if tgt_ext in valid_image_exts:
        ext = tgt_ext
    elif tgt_ext in valid_video_exts:
        ext = tgt_ext
    else:
        ext = '.jpg'
    out_name = f'output_{ts}{ext}'
    out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_name)

    result = process_image_swap(src_path, tgt_path, out_path,
                                face_indices=face_indices,
                                enhance=enhance,
                                pitch_semitones=pitch)
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
          <span style="font-size:.9rem">Enable GFPGAN enhancement</span>
        </label>
      </div>
      <div>
        <label>Audio Pitch Shift (video only)</label>
        <div class="range-row">
          <input type="range" id="pitchRange" min="-12" max="12" step="1" value="0"
                 oninput="document.getElementById('pitchVal').textContent=this.value">
          <span class="range-val" id="pitchVal">0</span>
          <span style="font-size:.8rem;color:var(--muted)">semitones</span>
        </div>
        <div style="font-size:.75rem;color:var(--muted);margin-top:4px">
          Negative = lower pitch, Positive = higher pitch (requires ffmpeg)
        </div>
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

// ── Generic helpers ───────────────────────────────────────────────────────────
function showAlert(elId, msg, type){
  const el = document.getElementById(elId);
  el.textContent = msg;
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
  previewFile(this, 'tgtPreview');
  document.getElementById('swapOutput').style.display='none';
  hideAlert('swapAlert');
});

// ── Swap tab ──────────────────────────────────────────────────────────────────
async function doSwap(){
  const src = document.getElementById('srcFile').files[0];
  const tgt = document.getElementById('tgtFile').files[0];
  if(!src || !tgt){ showAlert('swapAlert','Select both source and target files','warn'); return; }

  const btn = document.getElementById('swapBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Processing…';
  hideAlert('swapAlert');
  document.getElementById('swapOutput').style.display='none';

  const prog = document.getElementById('swapProgress');
  prog.style.display='block';
  let w=5; const barInter = setInterval(()=>{ w=Math.min(w+2,85); document.getElementById('swapBar').style.width=w+'%'; },400);

  const fd = new FormData();
  fd.append('source', src);
  fd.append('target', tgt);
  fd.append('enhance', document.getElementById('enhanceCheck').checked ? '1' : '0');
  fd.append('pitch_semitones', document.getElementById('pitchRange').value);

  try {
    const res = await fetch('/api/swap', {method:'POST', body:fd});
    const data = await res.json();
    clearInterval(barInter);
    document.getElementById('swapBar').style.width='100%';
    setTimeout(()=>prog.style.display='none',600);

    if(data.success){
      showAlert('swapAlert', `✅ Done in ${data.processing_time}s — ${data.faces_swapped} face(s) swapped`, 'success');
      renderOutput(data.output_filename, data.output_type, data.preview_b64);
    } else {
      showAlert('swapAlert', '❌ ' + (data.error || 'Swap failed'), 'error');
    }
  } catch(e){
    clearInterval(barInter); prog.style.display='none';
    showAlert('swapAlert', 'Network error: ' + e.message, 'error');
  } finally {
    btn.disabled=false; btn.innerHTML='⚡ Swap Faces';
  }
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
