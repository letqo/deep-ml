import numpy as np

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute the RBF (Gaussian) kernel matrix between X1 and X2.
    
    Args:
        X1: First set of samples with shape (n1, d)
        X2: Second set of samples with shape (n2, d)
        gamma: Kernel coefficient (controls kernel width)
    
    Returns:
        Kernel matrix of shape (n1, n2)
    """

    x1 = X1[:, None, :]
    x2 = X2[None, :, :]

    squared_diff = np.sum((x1 - x2)**2, axis=-1)
    kernel_dist = np.exp(- gamma * squared_diff)

    return np.round(kernel_dist, 4)
