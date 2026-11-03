import numpy as np

def bias_variance_decomp(predictions, y_true):
    """
    Compute the empirical bias-variance decomposition from bootstrap predictions.

    Args:
        predictions: array-like of shape (B, M) - predictions from B models at M test points
        y_true: array-like of shape (M,) - true target values

    Returns:
        dict with keys 'bias_squared', 'variance', 'mse'
    """
    predictions = np.array(predictions)
    y_true = np.array(y_true)
    y_bar = np.mean(predictions, axis=0)
    bias_squarred = np.mean((y_bar - y_true)**2)
    variance = np.mean(np.mean((predictions - y_bar)**2, axis=0))
    mse = np.mean(np.mean((predictions - y_true)**2, axis=0))

    return {
        "bias_squared" : bias_squarred,
        "variance" : variance,
        "mse" : mse
    }
