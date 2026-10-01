import numpy as np

def to_categorical(x, n_col=None):
	
	if n_col == None:
		n_col = np.max(x) + 1

	M = np.eye(n_col)
	
	return M[x]