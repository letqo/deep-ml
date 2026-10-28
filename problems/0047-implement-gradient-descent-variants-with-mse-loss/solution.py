import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    m, n = X.shape
    theta = np.array(weights)
 
    if method == 'stochastic':
        b = 1
    elif method == 'batch':
        b = m
    elif method == 'mini_batch':
        b = batch_size

    for _ in range(n_epochs):
        s = 0
        while s < m:
            X_B = X[s : s + b]
            y_B = y[s : s + b]
            hat_y = X_B @ theta
            hat_error = hat_y - y_B
            grad = (2 / len(X_B)) * X_B.T @ hat_error
            theta = theta - learning_rate * grad
            s += b
    
    return theta
