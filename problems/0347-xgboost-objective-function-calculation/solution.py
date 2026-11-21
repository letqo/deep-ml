import numpy as np

def xgboost_objective(gradients: np.ndarray, hessians: np.ndarray,
                      left_indices: np.ndarray, right_indices: np.ndarray,
                      lambda_reg: float = 1.0, gamma: float = 0.0) -> dict:
    """
    Calculate XGBoost objective function components for a potential split.
    
    Args:
        gradients: First-order gradients for each sample
        hessians: Second-order hessians for each sample
        left_indices: Indices of samples going to left child
        right_indices: Indices of samples going to right child
        lambda_reg: L2 regularization parameter
        gamma: Tree complexity penalty
        
    Returns:
        Dictionary with 'left_weight', 'right_weight', and 'gain'
    """

    gl = np.sum(gradients[left_indices])
    gr = np.sum(gradients[right_indices])
    hl = np.sum(hessians[left_indices])
    hr = np.sum(hessians[right_indices])
    
    weight_r = - gr/(hr + lambda_reg)
    weight_l = - gl/(hl + lambda_reg)

    score_r = gr **2 /(hr + lambda_reg)
    score_l = gl **2 /(hl + lambda_reg)
    score_parent = (gl + gr)**2 /(hl + hr + lambda_reg)

    gain = 1/2 * (score_r + score_l - score_parent) - gamma

    return {
        'left_weight': round(weight_l, 4),
        'right_weight': round(weight_r, 4),
        'gain': round(gain, 4)
    }