"""
Landmark stabilizer.

Applies a per-coordinate 1-Euro filter to the 5-point face landmarks across
frames, so the alignment used by the swapper stops jittering. Faces are matched
across frames by nearest centroid (single-face clips collapse to one track).

Usage (only when DEEPTRACE_TEMPORAL=1):

    stabilizer = get_stabilizer()
    smoothed = stabilizer.stabilize(landmark_5, frame_index)

`stabilize` returns a NEW array; it never mutates the input.
"""

import threading
from typing import List, Optional

import numpy

from deeptrace.research.temporal.one_euro import OneEuroFilter

# 1-Euro parameters tuned for landmark coordinates in pixel units.
# Lower min_cutoff -> stronger smoothing when still; beta -> how fast it follows motion.
# Tuned on jittered-landmark simulations: ~42% frame-to-frame jitter reduction
# on a static face for <0.01px extra lag when the face is moving (see paper
# Table: temporal parameter sweep).
_MIN_CUTOFF = 0.3
_BETA = 0.01
_D_CUTOFF = 1.0

# Re-associate a face to a track only if its centroid is within this many pixels
# of the track's last centroid; otherwise start a fresh track.
_ASSOC_MAX_DISTANCE = 120.0

# Drop a track that has not been seen for this many frames.
_TRACK_TIMEOUT = 8


class _Track:
	def __init__(self, num_points : int) -> None:
		# One filter per coordinate (x and y of every landmark point).
		self.filters = [
			(OneEuroFilter(_MIN_CUTOFF, _BETA, _D_CUTOFF), OneEuroFilter(_MIN_CUTOFF, _BETA, _D_CUTOFF))
			for _ in range(num_points)
		]
		self.last_centroid : Optional[numpy.ndarray] = None
		self.last_frame : int = -1

	def apply(self, landmark : numpy.ndarray, frame_index : int) -> numpy.ndarray:
		smoothed = numpy.empty_like(landmark, dtype = numpy.float32)
		for i, (x, y) in enumerate(landmark):
			fx, fy = self.filters[i]
			smoothed[i, 0] = fx.filter(float(x), frame_index)
			smoothed[i, 1] = fy.filter(float(y), frame_index)
		self.last_centroid = smoothed.mean(axis = 0)
		self.last_frame = frame_index
		return smoothed


class LandmarkStabilizer:
	def __init__(self) -> None:
		self._tracks : List[_Track] = []
		self._lock = threading.Lock()

	def reset(self) -> None:
		with self._lock:
			self._tracks = []

	def stabilize(self, landmark_5 : numpy.ndarray, frame_index : int) -> numpy.ndarray:
		if landmark_5 is None:
			return landmark_5
		landmark = numpy.asarray(landmark_5, dtype = numpy.float32)
		centroid = landmark.mean(axis = 0)

		with self._lock:
			self._evict_stale(frame_index)
			track = self._match_track(centroid, frame_index)
			if track is None:
				track = _Track(num_points = landmark.shape[0])
				self._tracks.append(track)
			return track.apply(landmark, frame_index)

	def _match_track(self, centroid : numpy.ndarray, frame_index : int) -> Optional[_Track]:
		best_track = None
		best_distance = _ASSOC_MAX_DISTANCE
		for track in self._tracks:
			# Never bind two faces from the same frame to one track.
			if track.last_centroid is None or track.last_frame == frame_index:
				continue
			distance = float(numpy.linalg.norm(track.last_centroid - centroid))
			if distance < best_distance:
				best_distance = distance
				best_track = track
		return best_track

	def _evict_stale(self, frame_index : int) -> None:
		self._tracks = [
			track for track in self._tracks
			if track.last_frame < 0 or (frame_index - track.last_frame) <= _TRACK_TIMEOUT
		]


_stabilizer : Optional[LandmarkStabilizer] = None


def get_stabilizer() -> LandmarkStabilizer:
	global _stabilizer
	if _stabilizer is None:
		_stabilizer = LandmarkStabilizer()
	return _stabilizer
