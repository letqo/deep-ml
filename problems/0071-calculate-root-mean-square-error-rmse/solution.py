
import numpy as np

def rmse(y_true, y_pred):
	if y_pred is None or y_true is None or y_pred.shape != y_true.shape:
		raise ValueError("Check the shape of your inputs")
	
	rmse_res = np.sqrt(np.mean((y_true - y_pred)**2))

	return round(rmse_res,3)
