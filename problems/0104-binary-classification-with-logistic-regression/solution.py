import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.

	Args:
		X: Input feature matrix (shape: N x D)
		weights: Model weights (shape: D)
		bias: Model bias

	Returns:
		Binary predictions (0 or 1)
	"""
	sigmoid = lambda x : 1/(1 + np.exp(- np.clip(x, -500, 500)))
	z = X@weights + bias
	binary_pred = sigmoid(z) >= 0.5
	return binary_pred.astype("int")

	