import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
  
	shuffled_indexes = []
	shuffled_indexes = np.random.default_rng(seed).permutation(len(data))
	train_end = int(len(data) * train_frac)
	training_set = data[shuffled_indexes[ : train_end]]

	validation_end = train_end + int(len(data) * validation_frac)
    validation_set = data[shuffled_indexes[train_end : validation_end]]

	test_set = data[shuffled_indexes[validation_end : ]]
    
	return [training_set, validation_set, test_set]