"""
Adaptive swap controller.

run_adaptive_swap() wraps the normal swap with a two-stage cascade:

    Stage 0  choose a starting pixel-boost resolution from the face-to-frame
             area ratio (bigger face -> higher resolution is worth it).
    Stage 1  measure ArcFace identity similarity of the result to the source;
             if it is below ID_TARGET and a higher resolution is available,
             swap once more at the next resolution and keep the better result.

The escalation is capped at one extra swap per face to bound the worst-case CPU
cost at 2x a single swap (only on hard frames).
"""

from typing import Callable, Optional, Tuple

import numpy

from deeptrace import state_manager
from deeptrace.types import Face, VisionFrame

# Candidate pixel-boost resolutions, smallest -> largest.
_RESOLUTION_LADDER = [256, 512, 768, 1024]

# Stage 0: face-area-ratio thresholds -> starting resolution index on the ladder.
_SMALL_FACE_RATIO = 0.06    # below this, start low (high res wasted on tiny faces)
_LARGE_FACE_RATIO = 0.20    # above this, start one rung higher

# Stage 1: escalate only if identity similarity to source is below this.
_ID_TARGET = 0.88

# Scoring gate: the ArcFace verification in Stage 1 costs ~one extra forward
# pass, so we only pay it when escalation could plausibly help. We skip it (and
# accept the Stage-0 result) when the detector is already confident about the
# face, since high-confidence faces almost never fail the identity check. This
# turns the closed loop from always-on into on-demand.
_DETECTOR_CONFIDENCE_GATE = 0.85

SwapFn = Callable[[Face, Face, VisionFrame, numpy.ndarray, Tuple[int, int]], VisionFrame]


def _face_area_ratio(target_face : Face, vision_frame : VisionFrame) -> float:
	box = target_face.bounding_box
	face_area = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
	frame_area = float(vision_frame.shape[0] * vision_frame.shape[1])
	return face_area / frame_area if frame_area > 0 else 0.0


def _ladder_index_for_face(target_face : Face, vision_frame : VisionFrame, configured : int) -> int:
	# Start from the configured resolution, then nudge by face size.
	try:
		base = _RESOLUTION_LADDER.index(configured)
	except ValueError:
		base = 0
	ratio = _face_area_ratio(target_face, vision_frame)
	if ratio < _SMALL_FACE_RATIO:
		base = max(0, base - 1)
	elif ratio > _LARGE_FACE_RATIO:
		base = min(len(_RESOLUTION_LADDER) - 1, base + 1)
	return base


def _detector_confidence(target_face : Face) -> float:
	try:
		return float(target_face.score_set.get('detector'))
	except Exception:
		return 0.0


def _score_identity(paste_frame : VisionFrame, target_face : Face, source_face : Face) -> Optional[float]:
	from deeptrace.research.metrics import cosine_similarity, swapped_embedding

	landmark_5 = target_face.landmark_set.get('5/68')
	swapped_norm = swapped_embedding(paste_frame, landmark_5)
	if swapped_norm is None or source_face.embedding_norm is None:
		return None
	return cosine_similarity(swapped_norm, source_face.embedding_norm)


def run_adaptive_swap(source_face : Face, target_face : Face, temp_vision_frame : VisionFrame, swap_landmark_5 : numpy.ndarray, configured_pixel_boost : Tuple[int, int], swap_fn : SwapFn) -> Tuple[VisionFrame, str]:
	"""
	Returns (paste_vision_frame, used_pixel_boost_string).
	"""
	configured_side = int(configured_pixel_boost[0])
	start_index = _ladder_index_for_face(target_face, temp_vision_frame, configured_side)

	start_side = _RESOLUTION_LADDER[start_index]
	result = swap_fn(source_face, target_face, temp_vision_frame, swap_landmark_5, (start_side, start_side))

	# Scoring gate: skip the expensive ArcFace verification when escalation is
	# impossible (already at max resolution) or unlikely to be needed (the
	# detector is confident about this face). This is what keeps the closed loop
	# cheap on clean footage.
	has_headroom = start_index < len(_RESOLUTION_LADDER) - 1
	is_uncertain = _detector_confidence(target_face) < _DETECTOR_CONFIDENCE_GATE
	if not has_headroom or not is_uncertain:
		return result, '{}x{}'.format(start_side, start_side)

	# Stage 1: only reached for low-confidence faces that can still escalate.
	score = _score_identity(result, target_face, source_face)
	if score is not None and score < _ID_TARGET:
		next_side = _RESOLUTION_LADDER[start_index + 1]
		candidate = swap_fn(source_face, target_face, temp_vision_frame, swap_landmark_5, (next_side, next_side))
		candidate_score = _score_identity(candidate, target_face, source_face)
		if candidate_score is not None and candidate_score > score:
			return candidate, '{}x{}'.format(next_side, next_side)

	return result, '{}x{}'.format(start_side, start_side)
