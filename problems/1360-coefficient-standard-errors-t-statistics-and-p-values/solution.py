import math

import numpy as np


def ols_inference(X: np.ndarray, y: np.ndarray) -> tuple:
    """Least-squares fit with an intercept, plus inference on each coefficient.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        tuple: (coef, se, t, p), each a length-(p+1) array, intercept first.
    """
    n, p = X.shape

    D = np.column_stack([np.ones(len(X)), X])
    #Coefficients
    beta_hat = np.linalg.inv(D.T @ D) @ D.T @ y
    #Predictions y_hat
    y_hat = D @ beta_hat
    #Residual sum of squares
    rss = np.sum((y - y_hat)**2)
    #Variance
    hat_variance_squa = rss/(n - p - 1)
    
    hat_var_beta_hat = hat_variance_squa * np.linalg.inv(D.T @ D)

    #SE(beta_hat)
    SE_beta_hat = np.sqrt(np.diag(hat_var_beta_hat))
    #t stat 
    t_stats = beta_hat/SE_beta_hat

    #phi function
    phi = lambda x : 1/2 * (1 + math.erf(x/np.sqrt(2)))
    #p- value
    p_val = np.array([2 * (1 - phi(np.abs(t_stat))) for t_stat in t_stats])

    return beta_hat, SE_beta_hat, t_stats, p_val

