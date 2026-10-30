
import numpy as np

def r_squared(y_true, y_pred):
	y_bar = np.mean(y_true)
	SSR = np.sum((y_true - y_pred)**2)
	SST = np.sum((y_true - y_bar)**2)

	R_squa = (1 - SSR/SST)
	return R_squa
