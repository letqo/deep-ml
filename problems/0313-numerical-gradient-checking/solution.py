import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    x_copy = x.astype(float).copy()
    numerical_grad = np.zeros_like(x_copy)
    for i in range(len(x_copy)):

        original = x_copy[i]
        x_copy[i] = original + epsilon
        f_plus = f(x_copy)
    
        x_copy[i] = original - epsilon
        f_minus = f(x_copy)
        numerical_grad[i] = (f_plus - f_minus)/(2 * epsilon)
    
    denom = np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad)

    if denom == 0:
        return numerical_grad, 0.0
    error = np.linalg.norm(numerical_grad - analytical_grad) / denom

    return (numerical_grad, error)



