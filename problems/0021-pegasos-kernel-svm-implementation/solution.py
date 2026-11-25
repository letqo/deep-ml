import numpy as np

def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:
    """
    Train a kernel SVM using the deterministic Pegasos algorithm.
    
    Args:
        data: Training data of shape (n_samples, n_features)
        labels: Labels of shape (n_samples,) with values in {-1, 1}
        kernel: 'linear' or 'rbf'
        lambda_val: Regularization parameter
        iterations: Number of training iterations
        sigma: RBF kernel bandwidth (only used if kernel='rbf')
    
    Returns:
        Tuple of (alphas, bias) where alphas is a list and bias is a float
    """
    def K(a, b):
        if kernel == 'linear':
            return a @ b
        else:
            squared_diff = np.sum((a - b)**2)
            kernel_dist = np.exp(- squared_diff/(2 * sigma **2))
            return kernel_dist

    n, p = data.shape
    alphas = np.zeros(n)
    b = 0.0
    for t in range(1, iterations + 1):
        eta = 1/(lambda_val * t)
        for i in range(n):
            yi = labels[i]
            gi = np.sum([alphas[j] * labels[j] * K(data[i], data[j]) for j in range(n)]) + b

            if yi * gi < 1:
                alphas[i] += eta * (1 - lambda_val * alphas[i])
                b += eta * yi
            else:
               alphas[i] += - eta * lambda_val * alphas[i]
    return np.round(alphas, 4), np.round(b, 4)

    