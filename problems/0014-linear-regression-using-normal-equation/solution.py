import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X_trans = np.transpose(X)
	X_T_X = X_trans @ X
	X_T_X_inv = np.linalg.inv(X_T_X)
	theta = X_T_X_inv @ X_trans @ y
	return np.round(theta, 4) 