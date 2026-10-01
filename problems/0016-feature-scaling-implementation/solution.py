import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	
	standardized_data = (data - mean)/std

	maxi = np.max(data, axis=0)
	mini = np.min(data, axis=0)

	normalized_data = (data - mini)/(maxi - mini)

	return standardized_data, normalized_data