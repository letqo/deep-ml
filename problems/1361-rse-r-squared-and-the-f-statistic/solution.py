import numpy as np


def model_fit_stats(X: np.ndarray, y: np.ndarray) -> tuple:
    """Residual standard error, R-squared and the overall F-statistic.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        tuple: (rse, r2, f_stat) as floats. RSS == 0 gives (0.0, 1.0, inf).
    """
    n, p = X.shape
    D = np.column_stack([np.ones(n), X])

    beta_hat = np.linalg.inv(D.T@D) @ D.T@y
    y_hat = D@beta_hat
    y_mean = np.mean(y)

    rss = np.sum((y - y_hat)**2)

    if np.isclose(rss, 0):
        return 0.0, 1.0, float('inf')

    rse = np.sqrt(rss/(n - p - 1))
    tss = np.sum((y - y_mean)**2)
    r2 = 1 - rss/tss
    f_stat = ((tss - rss)/p)/(rss/(n - p - 1))

    return float(rse), float(r2), float(f_stat)
