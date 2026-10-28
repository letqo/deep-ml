
def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	dot_pro = 0
	norm_L = 0
	for i in range(len(v)):
		dot_pro += v[i] * L[i]
		norm_L += L[i] * L[i]
	
	v_projected = [(dot_pro/norm_L) * i for i in L]

	return v_projected
