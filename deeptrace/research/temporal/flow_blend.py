"""
Motion-compensated temporal blending of the swapped output (Contribution A, part 2).

Landmark stabilization (stabilizer.py) removes geometric jitter, but most of it
is absorbed by the swapper's own face alignment, so appearance-level flicker
(texture shimmer between frames) survives. This module attacks that directly:

    1. estimate dense optical flow between the previous and current TARGET frame
    2. warp the previous SWAPPED output along that flow into the current frame
    3. blend the warped previous output with the current output, but only where
       the flow is reliable (forward-backward consistency) and only inside the
       changed (face) region.

The result is an exponential moving average in motion-compensated space: the
swapped texture becomes coherent across frames instead of shimmering, without
smearing on fast motion (low-confidence flow falls back to the current output).

Enabled by DEEPTRACE_FLOW_BLEND=1. Disabled -> the current output is returned
untouched (stock behavior).
"""

import threading
from typing import Optional

import cv2
import numpy

# Weight of the (motion-compensated) previous output where flow is reliable.
# 0 -> no blending (stock); higher -> smoother but more lag on fast motion.
_BLEND_PREV = 0.45
# Forward-backward flow disagreement (px) above which a pixel is "unreliable".
_FB_THRESHOLD = 1.5
# Only blend where the swap actually changed the frame by more than this (per
# channel, 0-255) so the untouched background is never altered.
_CHANGE_THRESHOLD = 2.0


class FlowBlender:
	def __init__(self) -> None:
		self._prev_target_gray : Optional[numpy.ndarray] = None
		self._prev_output : Optional[numpy.ndarray] = None
		self._prev_frame_index : Optional[int] = None
		self._flow = _create_flow()
		self._lock = threading.Lock()

	def reset(self) -> None:
		with self._lock:
			self._prev_target_gray = None
			self._prev_output = None
			self._prev_frame_index = None

	def blend(self, target_frame : numpy.ndarray, output_frame : numpy.ndarray, frame_index : Optional[int]) -> numpy.ndarray:
		"""
		target_frame : the ORIGINAL (pre-swap) frame, used only to estimate motion.
		output_frame : the swapped frame to be temporally smoothed.
		Returns a new frame; inputs are not mutated.
		"""
		with self._lock:
			gray = cv2.cvtColor(target_frame, cv2.COLOR_RGB2GRAY)
			result = output_frame

			# Only blend consecutive frames; gaps / out-of-order resets the state.
			contiguous = (
				self._prev_output is not None
				and self._prev_frame_index is not None
				and frame_index is not None
				and frame_index == self._prev_frame_index + 1
			)
			if contiguous:
				result = self._blend_with_prev(gray, output_frame, target_frame)

			self._prev_target_gray = gray
			self._prev_output = result.astype(numpy.float32)
			self._prev_frame_index = frame_index
			return result

	def _blend_with_prev(self, gray : numpy.ndarray, output_frame : numpy.ndarray, target_frame : numpy.ndarray) -> numpy.ndarray:
		flow_fw = self._flow(self._prev_target_gray, gray)   # prev -> cur
		flow_bw = self._flow(gray, self._prev_target_gray)   # cur -> prev

		warped_prev = _warp(self._prev_output, flow_bw)

		# Reliability: forward-backward flow consistency.
		warped_fw = _warp(flow_fw, flow_bw)
		fb_diff = numpy.linalg.norm(flow_bw + warped_fw, axis = 2)
		reliable = fb_diff < _FB_THRESHOLD

		# Region the swap actually changed (keep background pixel-identical).
		changed = numpy.max(numpy.abs(output_frame.astype(numpy.float32) - target_frame.astype(numpy.float32)), axis = 2) > _CHANGE_THRESHOLD

		weight = numpy.where(reliable & changed, _BLEND_PREV, 0.0).astype(numpy.float32)[..., None]
		blended = (1.0 - weight) * output_frame.astype(numpy.float32) + weight * warped_prev
		return numpy.clip(blended, 0, 255).astype(output_frame.dtype)


def _create_flow():
	if hasattr(cv2, 'DISOpticalFlow_create'):
		dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
		return lambda a, b: dis.calc(a, b, None)
	return lambda a, b: cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def _warp(image : numpy.ndarray, flow : numpy.ndarray) -> numpy.ndarray:
	h, w = flow.shape[:2]
	grid_x, grid_y = numpy.meshgrid(numpy.arange(w), numpy.arange(h))
	map_x = (grid_x + flow[..., 0]).astype(numpy.float32)
	map_y = (grid_y + flow[..., 1]).astype(numpy.float32)
	return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode = cv2.BORDER_REPLICATE)


_blender : Optional[FlowBlender] = None


def get_blender() -> FlowBlender:
	global _blender
	if _blender is None:
		_blender = FlowBlender()
	return _blender
