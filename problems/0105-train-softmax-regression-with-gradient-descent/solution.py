import numpy as np

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss.
	"""
	losses = []
	num_class = np.max(y) + 1
	n, p = X.shape
	softmax = lambda x : np.array(np.exp(x - np.max(x))/np.sum(np.exp(x - np.max(x))))
	#Initialization
	W = np.zeros((p + 1, num_class))
	#Adding bias column to X
	D = np.column_stack([np.ones(n), X])
	Y = np.eye(num_class, num_class)[y]
	for _ in range(iterations):
		#Z, the logits
		Z = D@W
		#Prediction
		P = np.array([softmax(row) for row in Z])
		#Loss
		losses.append(-np.sum(Y * np.log(P)))
		#Update
		W = W - learning_rate * D.T@(P - Y)

	return np.round(W.T, 4), losses
