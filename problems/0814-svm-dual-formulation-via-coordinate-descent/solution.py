import numpy as np

def svm_dual_coordinate_descent(X, y, C=1.0, n_iters=1000, tol=1e-9):
    """
    Solve the soft-margin linear SVM dual via coordinate descent.

    Args:
        X: (n_samples, n_features) feature matrix
        y: (n_samples,) labels in {-1, +1}
        C: regularization parameter
        n_iters: maximum number of passes over the data
        tol: convergence tolerance on alpha changes

    Returns:
        List of final alpha values (length n_samples).
    """
    x = np.array(X)
    y = np.array(y)
    
    n, p = x.shape
    biggest_change = 0
    w = np.zeros(p)
    alphas = np.zeros(n)

    for _ in range(n_iters):
        for i in range(n):
            margin = y[i] * (w @ x[i])
            leng2 = x[i] @ x[i]
            new_alpha = np.clip(alphas[i] - (margin - 1)/leng2, 0, C)
            change = new_alpha - alphas[i]
            w += (change) * y[i] * x[i]
            if biggest_change < np.abs(change):
                biggest_change = np.abs(change)
            alphas[i] = new_alpha
        change = 0
        if biggest_change < tol:
            break

    return alphas

