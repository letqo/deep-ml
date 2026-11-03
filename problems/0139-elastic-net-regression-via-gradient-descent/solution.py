import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    n_samples, n_features = X.shape
    #Initialization 
    weights = np.zeros(n_features)
    bias = 0
    for _ in range(max_iter):
        #Make prediction
        y_hat = X @ weights + bias
        #Compute residuals
        residuals = y_hat - y
        #Update weights and bias using gradients 
        ##Grad with respect to weights (is a vector)
        grad_weights = 1/n_samples * (X.T @ (y_hat - y)) + alpha1 * np.sign(weights) + 2 * alpha2 * weights
        ##Grad with respect to bias (is a scalar)
        grad_bias = np.mean(y_hat - y)
        #Update rule
        weights = weights - learning_rate * grad_weights
        bias = bias - learning_rate * grad_bias
        #Check for convergence
        L1_norm_weigths_grad = np.linalg.norm(grad_weights, ord=1)
        if L1_norm_weigths_grad < tol:
            break

    return weights, bias


