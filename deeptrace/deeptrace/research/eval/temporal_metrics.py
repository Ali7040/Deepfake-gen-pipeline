"""
Temporal-consistency metrics.

warping_error(video):
    Optical-flow "warping error" (Lai et al., ECCV 2018). For each adjacent
    frame pair it warps the previous frame into the current one using dense
    optical flow and measures the residual over non-occluded pixels (occlusion
    found by forward-backward flow consistency). LOWER = less flicker = more
    temporally consistent. Needs only OpenCV and the output video.

identity_consistency(rows):
    From the per-frame swapped-face embeddings logged by the instrumentation:
        TL-ID  mean cosine similarity between CONSECUTIVE frames (local jitter)
        TG-ID  mean cosine similarity of each frame to the clip's mean identity
    HIGHER = more stable identity across the clip.
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy


def _create_flow():
	# DIS optical flow is fast on CPU; fall back to Farneback if unavailable.
	if hasattr(cv2, 'DISOpticalFlow_create'):
		flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
		return lambda a, b: flow.calc(a, b, None)
	return lambda a, b: cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def _warp(image : numpy.ndarray, flow : numpy.ndarray) -> numpy.ndarray:
	h, w = flow.shape[:2]
	grid_x, grid_y = numpy.meshgrid(numpy.arange(w), numpy.arange(h))
	map_x = (grid_x + flow[..., 0]).astype(numpy.float32)
	map_y = (grid_y + flow[..., 1]).astype(numpy.float32)
	return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode = cv2.BORDER_REPLICATE)


def warping_error(video_path : str, max_frames : Optional[int] = None, fb_threshold : float = 1.5) -> Dict[str, Any]:
	capture = cv2.VideoCapture(video_path)
	if not capture.isOpened():
		raise IOError('Could not open video: ' + video_path)

	calc_flow = _create_flow()
	errors : List[float] = []
	prev_gray = None
	prev_color = None
	frame_count = 0

	while True:
		ok, frame = capture.read()
		if not ok or (max_frames is not None and frame_count >= max_frames):
			break
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		color = frame.astype(numpy.float32)

		if prev_gray is not None:
			# Predict current frame from previous via backward flow (cur -> prev).
			flow_bw = calc_flow(gray, prev_gray)
			flow_fw = calc_flow(prev_gray, gray)
			warped_prev = _warp(prev_color, flow_bw)

			# Occlusion mask: forward-backward flow consistency.
			warped_fw = _warp(flow_fw, flow_bw)
			fb_diff = numpy.linalg.norm(flow_bw + warped_fw, axis = 2)
			valid = (fb_diff < fb_threshold).astype(numpy.float32)

			if valid.sum() > 0:
				residual = numpy.mean((color - warped_prev) ** 2, axis = 2)
				masked_error = float((residual * valid).sum() / valid.sum())
				errors.append(masked_error)

		prev_gray = gray
		prev_color = color
		frame_count += 1

	capture.release()

	if not errors:
		return { 'warping_error': None, 'frame_pairs': 0 }
	return {
		'warping_error': round(float(numpy.mean(errors)), 4),
		'warping_error_per_frame': [round(e, 4) for e in errors],
		'frame_pairs': len(errors),
	}


def identity_consistency(rows : List[Dict[str, Any]]) -> Dict[str, Any]:
	ordered = sorted(
		[r for r in rows if r.get('embedding') is not None],
		key = lambda r: (r.get('frame_index') if r.get('frame_index') is not None else 0)
	)
	embeddings = [numpy.asarray(r['embedding'], dtype = numpy.float32) for r in ordered]
	if len(embeddings) < 2:
		return { 'tl_id': None, 'tg_id': None, 'frames': len(embeddings) }

	def cosine(a, b):
		denom = float(numpy.linalg.norm(a) * numpy.linalg.norm(b))
		return float(numpy.dot(a, b) / denom) if denom > 1e-8 else 0.0

	tl = [cosine(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]
	mean_embedding = numpy.mean(embeddings, axis = 0)
	tg = [cosine(e, mean_embedding) for e in embeddings]

	return {
		'tl_id': round(float(numpy.mean(tl)), 5),
		'tg_id': round(float(numpy.mean(tg)), 5),
		'frames': len(embeddings),
	}
