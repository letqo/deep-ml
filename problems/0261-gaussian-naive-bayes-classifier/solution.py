import numpy as np

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
	"""
	Implements Gaussian Naive Bayes classifier.
	
	Args:
		X_train: Training features (shape: N_train x D)
		y_train: Training labels (shape: N_train)
		X_test: Test features (shape: N_test x D)
	
	Returns:
		Predicted class labels for X_test (shape: N_test)
	"""
	epsilon = 1e-9
	classes = np.unique(y_train)
	prior, mean, variance = [], [], []
	for c in classes:
		mask = y_train == c
		c_count = sum(mask)
		prior.append(c_count/len(y_train))
		
		mean.append(np.mean(X_train[mask], axis=0))
		variance.append(np.var(X_train[mask], axis=0) + epsilon)

	mean = np.array(mean)
	variance = np.array(variance)
	prior = np.array(prior)

	scores = []
	#Prediction
	for idx, _ in enumerate(classes):
		distances = -1/2 * np.log(2 * np.pi * variance[idx]) - (X_test - mean[idx])**2/(2 * variance[idx]) 
		score = np.sum(distances, axis=1) + np.log(prior[idx])
		scores.append(score)

	return classes[np.argmax(scores, axis=0)]
