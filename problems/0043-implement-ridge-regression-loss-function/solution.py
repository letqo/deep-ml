import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	y_hat = X @ w 
	mean_SSR = np.mean((y_true - y_hat)**2)
	loss = mean_SSR + alpha * np.sum(w**2)

	return loss
