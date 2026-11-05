import numpy as np

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	softmax = np.exp(np.array(logits))/np.sum( np.exp(np.array(logits)))
	M = np.eye(len(logits))
	y = M[target]
	gradient = softmax - y
	return gradient
	