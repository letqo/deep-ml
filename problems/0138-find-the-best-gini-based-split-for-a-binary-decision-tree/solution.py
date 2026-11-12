import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    
    best_feature = float('inf')
    best_threshold = float('inf')
    best_score = float('inf')
    n, d = X.shape

    for j in range(d):
        col = X[:, j]
        thresholds = np.unique(col)
        for threshold in thresholds:

            mask = col <= threshold
            left = y[mask]
            right = y[~mask]

            gini_left = gini_score(left)
            gini_right = gini_score(right)

            weighted_score = len(left)/n * gini_left + len(right)/n * gini_right
            
            if best_score > weighted_score:
                best_feature = j
                best_threshold = threshold
                best_score = weighted_score

    return best_feature, best_threshold

def gini_score(y: np.ndarray):
    if not len(y):
        return 0
    else:
        _, counts = np.unique(y, return_counts=True)
        fract = counts/len(y)
        gini = 1 - np.sum(fract **2)

    return gini

