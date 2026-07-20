"""
Read-only quality metrics.

These functions only MEASURE frames; they never modify them. They are the
ground truth used to compare the stock baseline against the temporal /
adaptive pipelines in the ablation study.

    identity_similarity  - ArcFace cosine similarity to the source identity
                           (the headline "ID-Sim" metric)
    sharpness            - variance of Laplacian, a no-reference sharpness proxy
"""

from typing import Optional

import numpy

from deeptrace.types import Face, FaceLandmark5, VisionFrame


def cosine_similarity(embedding_a : numpy.ndarray, embedding_b : numpy.ndarray) -> float:
	a = embedding_a.ravel().astype(numpy.float32)
	b = embedding_b.ravel().astype(numpy.float32)
	denom = float(numpy.linalg.norm(a) * numpy.linalg.norm(b))
	if denom < 1e-8:
		return 0.0
	return float(numpy.dot(a, b) / denom)


def swapped_embedding(swapped_frame : VisionFrame, target_landmark_5 : FaceLandmark5) -> Optional[numpy.ndarray]:
	"""
	Normalised ArcFace embedding of the swapped face. Returns None on failure so
	callers can skip instead of crashing the pipeline.
	"""
	# Imported lazily to avoid any import-time coupling with the stock modules.
	from deeptrace import face_recognizer

	try:
		_, swapped_norm = face_recognizer.calculate_face_embedding(swapped_frame, target_landmark_5)
		return swapped_norm
	except Exception:
		return None


def identity_similarity(swapped_frame : VisionFrame, target_landmark_5 : FaceLandmark5, source_face : Face) -> Optional[float]:
	"""
	Cosine similarity between the swapped face's ArcFace embedding and the
	source identity. Higher is better (the swap looks more like the source).
	"""
	swapped_norm = swapped_embedding(swapped_frame, target_landmark_5)
	source_embedding = source_face.embedding_norm
	if swapped_norm is None or source_embedding is None:
		return None
	return cosine_similarity(swapped_norm, source_embedding)


def sharpness(vision_frame : VisionFrame) -> float:
	"""Variance of the Laplacian - a cheap no-reference sharpness/blur proxy."""
	import cv2

	gray = cv2.cvtColor(vision_frame, cv2.COLOR_RGB2GRAY) if vision_frame.ndim == 3 else vision_frame
	return float(cv2.Laplacian(gray, cv2.CV_64F).var())
