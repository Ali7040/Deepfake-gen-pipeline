"""
Per-swap metric recorder.

When DEEPTRACE_INSTRUMENT=1, the face swapper calls record_swap() once per
swapped face. Rows are buffered in memory and flushed to a JSONL file (one
JSON object per line) on process exit. Each row captures the metrics needed
for the baseline-vs-improved comparison.

Safety contract:
    record_swap() is wrapped so that ANY internal failure is swallowed and
    logged - instrumentation must never be able to crash or alter the swap.
"""

import atexit
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from deeptrace.research.config import research_config
from deeptrace.research.metrics import cosine_similarity, sharpness, swapped_embedding
from deeptrace.types import Face, FaceLandmark5, VisionFrame

_lock = threading.Lock()
_rows : List[Dict[str, Any]] = []
_flushed = False


def is_enabled() -> bool:
	return research_config.instrument


def record_swap(
	source_face : Face,
	target_face : Face,
	swapped_frame : VisionFrame,
	*,
	model_name : Optional[str] = None,
	pixel_boost : Optional[str] = None,
	elapsed_ms : Optional[float] = None,
	frame_index : Optional[int] = None,
) -> None:
	"""Record one swap. Read-only: never mutates any argument."""
	if not research_config.instrument:
		return
	try:
		target_landmark_5 = target_face.landmark_set.get('5/68')
		# Compute the swapped embedding once: reused for ID-Sim (vs source) and,
		# offline, for temporal identity consistency (TL-ID / TG-ID).
		swapped_norm = swapped_embedding(swapped_frame, target_landmark_5)
		source_norm = source_face.embedding_norm
		id_sim = None
		if swapped_norm is not None and source_norm is not None:
			id_sim = cosine_similarity(swapped_norm, source_norm)
		row : Dict[str, Any] = {
			'ts': time.time(),
			'frame_index': frame_index,
			'model_name': model_name,
			'pixel_boost': pixel_boost,
			'elapsed_ms': round(elapsed_ms, 3) if elapsed_ms is not None else None,
			'id_sim': round(id_sim, 5) if id_sim is not None else None,
			'sharpness': round(sharpness(swapped_frame), 3),
			'target_score': _safe_detector_score(target_face),
			# Rounded embedding kept for offline temporal-identity metrics.
			'embedding': [round(float(v), 5) for v in swapped_norm] if swapped_norm is not None else None,
		}
		with _lock:
			_rows.append(row)
	except Exception as exception:  # pragma: no cover - instrumentation must never break the run
		from deeptrace.logger import warn
		warn('Instrumentation failed: ' + str(exception), __name__)


def _safe_detector_score(face : Face) -> Optional[float]:
	try:
		return round(float(face.score_set.get('detector')), 5)
	except Exception:
		return None


def flush() -> None:
	global _flushed
	with _lock:
		if _flushed or not _rows:
			return
		path = research_config.metrics_path
		directory = os.path.dirname(path)
		if directory:
			os.makedirs(directory, exist_ok = True)
		with open(path, 'w', encoding = 'utf-8') as metrics_file:
			for row in _rows:
				metrics_file.write(json.dumps(row) + '\n')
		_flushed = True


atexit.register(flush)
