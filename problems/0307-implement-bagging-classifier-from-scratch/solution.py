import numpy as np

def bagging_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_estimators: int = 10, seed: int = 42) -> np.ndarray:
    """
    Implement a bagging classifier using decision stumps.
    
    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,), binary {0, 1}
        X_test: Test features of shape (n_test_samples, n_features)
        n_estimators: Number of bootstrap samples/base estimators
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Predicted labels for X_test
    """
    np.random.seed(seed)
    n = len(y_train)
    all_predictions = []

    for _ in range(n_estimators):
        idx = np.random.choice(n, size=n, replace=True)
        X_boot, y_boot = X_train[idx], y_train[idx]
        stump = fit_stump(X_boot, y_boot)
        all_predictions.append(predict_stump(X_test, stump))
        
    vote = np.mean(all_predictions, axis=0)

    return (vote >= 0.5).astype(int)
    

def fit_stump(X, y):
    best_errors = float('inf')
    best_stump = None

    for j in range(X.shape[1]):
        for t in np.unique(X[:, j]):
            for direction in (1, 2):
                # predictions = X[:, j] > t if direction == 1 else X[:, j] <= t
                errors = np.sum(y != predict_stump(X, (j, t, direction)))
                if errors < best_errors:           
                    best_errors = errors
                    best_stump = (j, t, direction)

    return best_stump

def predict_stump(X, stump):
    # Part 2: use a stump to predict
    feature, threshold, direction = stump
    predictions = X[:, feature] > threshold if direction == 1 else X[:, feature] <= threshold

    return predictions.astype(int)

    