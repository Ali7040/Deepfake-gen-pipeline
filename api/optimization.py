"""
Runtime optimizations for DeepTrace inference.

Auto-detects CUDA, configures threading, and tunes ORT session options
for the fastest possible inference on the available hardware.
"""
from __future__ import annotations

import os
import logging
from typing import List, Tuple

logger = logging.getLogger("deeptrace.api.optimization")


def detect_best_providers() -> Tuple[List[str], str]:
    """
    Pick the best execution providers ordered by speed.
    Returns (providers, description).
    """
    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
    except Exception as e:
        logger.warning("onnxruntime import failed: %s", e)
        return ['CPUExecutionProvider'], 'CPU (onnxruntime unavailable)'

    providers: List[str] = []
    if 'CUDAExecutionProvider' in available:
        providers.append('CUDAExecutionProvider')
    if 'DmlExecutionProvider' in available:  # Windows DirectML
        providers.append('DmlExecutionProvider')
    if 'CoreMLExecutionProvider' in available:  # macOS
        providers.append('CoreMLExecutionProvider')
    providers.append('CPUExecutionProvider')

    desc = ' → '.join(p.replace('ExecutionProvider', '') for p in providers)
    return providers, desc


def detect_thread_count() -> int:
    """Use up to 8 threads, capped at physical core count."""
    try:
        cores = os.cpu_count() or 4
    except Exception:
        cores = 4
    return min(8, max(2, cores))


def apply_optimizations() -> dict:
    """
    Override DeepTrace state with hardware-tuned values.
    Call AFTER initialize_deeptrace().
    Returns a dict describing what was changed (for logging / health endpoint).
    """
    from deeptrace import state_manager

    providers, desc = detect_best_providers()
    threads = detect_thread_count()
    has_gpu = providers[0] != 'CPUExecutionProvider'

    state_manager.set_item('execution_providers', providers)
    state_manager.set_item('execution_thread_count', threads)

    # On GPU we can afford a bigger detector and full enhancer; on CPU keep lean.
    if has_gpu:
        state_manager.set_item('face_detector_size', '640x640')
        state_manager.set_item('face_enhancer_blend', 100)
        state_manager.set_item('video_memory_strategy', 'tolerant')
    else:
        state_manager.set_item('face_detector_size', '640x640')
        state_manager.set_item('face_enhancer_blend', 80)
        state_manager.set_item('video_memory_strategy', 'moderate')

    # ORT threading hints (read by inference_manager when sessions are created)
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['ORT_INTRA_OP_NUM_THREADS'] = str(threads)
    os.environ['ORT_INTER_OP_NUM_THREADS'] = '1'

    summary = {
        'providers': providers,
        'providers_desc': desc,
        'thread_count': threads,
        'gpu_enabled': has_gpu,
    }
    logger.info("DeepTrace optimizations applied: %s", summary)
    return summary
