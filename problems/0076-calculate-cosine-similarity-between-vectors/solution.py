import numpy as np
from math import sqrt


def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	dot = sum([x*y for x,y in zip(v1,v2)])
	v1_norm = sqrt(sum([x**2 for x in v1]))
	v2_norm = sqrt(sum([x**2 for x in v2]))

	return dot / (v1_norm * v2_norm)