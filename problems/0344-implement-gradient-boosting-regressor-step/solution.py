import numpy as np

def gradient_boosting_step(X, y, current_predictions, learning_rate=0.1):
    """
    Perform one step of gradient boosting regression using a decision stump.
    
    Args:
        X: Feature matrix (list of lists), shape (n_samples, n_features)
        y: Target values (list), shape (n_samples,)
        current_predictions: Current ensemble predictions (list), shape (n_samples,)
        learning_rate: Learning rate for the update (default 0.1)
    
    Returns:
        List of updated predictions rounded to 4 decimal places
    """
    X = np.array(X)
    y = np.array(y)
    current_predictions = np.array(current_predictions)

    best_error = float('inf')
    best_feature, best_threshold, best_left, best_right = None, None, None, None

    n, p = X.shape

    res = y - current_predictions
    
    for j in range(p):
        col = X[:, j]
        uniq_val = np.unique(col)
        for l, k in zip(uniq_val[:-1], uniq_val[1:]):
            threshold = (l+k)/2
            mask = col <= threshold
            l_mean = np.mean(res[mask])
            r_mean = np.mean(res[~mask])
            l_avg_res = np.sum((res[mask] - l_mean) **2)
            r_avg_res = np.sum((res[~mask] - r_mean) **2)
            mse = r_avg_res + l_avg_res

            if mse < best_error:
                best_error = mse
                best_feature = j
                best_threshold = threshold
                best_left = l_mean
                best_right = r_mean

    if best_feature is None:
        predictions = np.full(n, np.mean(res))
    else:
        mask = X[:, best_feature] <= best_threshold
        predictions = np.where(mask, best_left, best_right)

    current_predictions = current_predictions + learning_rate * predictions

    return np.round(current_predictions, 4).tolist()


