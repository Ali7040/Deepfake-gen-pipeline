"""
1-Euro filter (Casiez, Roussel & Vogel, CHI 2012).

A speed-adaptive low-pass filter: it smooths heavily when the signal is nearly
still (removing jitter) and loosens when the signal moves fast (avoiding lag).
Ideal for landmark coordinates, which sit still then move quickly.

We index updates by integer frame number rather than wall-clock time so the
result is deterministic and reproducible across runs (important for a paper).
"""

import math
from typing import Optional


class _LowPassFilter:
	def __init__(self) -> None:
		self._value : Optional[float] = None

	def has_value(self) -> bool:
		return self._value is not None

	def last(self) -> Optional[float]:
		return self._value

	def filter(self, value : float, alpha : float) -> float:
		if self._value is None:
			self._value = value
		else:
			self._value = alpha * value + (1.0 - alpha) * self._value
		return self._value

	def reset(self) -> None:
		self._value = None


class OneEuroFilter:
	"""
	Scalar 1-Euro filter.

	Parameters
	----------
	min_cutoff : lower -> smoother but more lag when still.
	beta       : higher -> follows fast motion more tightly (less lag).
	d_cutoff   : cutoff for the derivative estimate.
	"""

	def __init__(self, min_cutoff : float = 1.0, beta : float = 0.0, d_cutoff : float = 1.0) -> None:
		self.min_cutoff = float(min_cutoff)
		self.beta = float(beta)
		self.d_cutoff = float(d_cutoff)
		self._x = _LowPassFilter()
		self._dx = _LowPassFilter()
		self._last_frame : Optional[int] = None

	@staticmethod
	def _alpha(cutoff : float, dt : float) -> float:
		tau = 1.0 / (2.0 * math.pi * cutoff)
		return 1.0 / (1.0 + tau / dt)

	def reset(self) -> None:
		self._x.reset()
		self._dx.reset()
		self._last_frame = None

	def filter(self, value : float, frame_index : int) -> float:
		# dt in "frames"; clamped to >= 1 so re-runs / duplicate indices stay stable.
		if self._last_frame is None:
			dt = 1.0
		else:
			dt = max(1.0, float(frame_index - self._last_frame))
		self._last_frame = frame_index

		prev = self._x.last()
		dvalue = 0.0 if prev is None else (value - prev) / dt
		edvalue = self._dx.filter(dvalue, self._alpha(self.d_cutoff, dt))

		cutoff = self.min_cutoff + self.beta * abs(edvalue)
		return self._x.filter(value, self._alpha(cutoff, dt))
