import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	n, p = X.shape
	sigmoid = lambda x : 1/(1 + np.exp(- np.clip(x, -500, 500)))
	
	D = np.column_stack([np.ones(n), X])
	losses = []
	#initialize to zero
	weights = np.zeros(p + 1)
	for _ in range(iterations):
		#predictions
		y_hat = sigmoid(D@weights)
		#the loss
		loss = - np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
		losses.append(loss)
		#update
		weights = weights - learning_rate * D.T@(y_hat - y)
	return weights, losses
