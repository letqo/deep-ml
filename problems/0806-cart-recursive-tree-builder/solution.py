import numpy as np

def build_cart_tree(X, y, max_depth=3, min_samples_split=2, depth=0):
    """
    Build a CART classification tree using Gini impurity.

    Returns a nested dict representing the tree.
    """
    def build(X, y, depth):
            # Part A: the leaf this node becomes if it stops here
            leaf = {'leaf': True, 'prediction': int(np.bincount(y).argmax())}

            # Part B: stopping conditions
            if len(np.unique(y)) == 1 or len(y) < min_samples_split or depth >= max_depth:
                return leaf

            feature, threshold = best_split(X, y)
            if feature is None:
                return leaf

            # Part C: split and recurse
            mask = X[:, feature] <= threshold
            left_tree = build(X[mask], y[mask], depth + 1)
            right_tree = build(X[~mask], y[~mask], depth + 1)

            return {'leaf': False, 'feature': int(feature), 'threshold': float(threshold),
                    'left': left_tree, 'right': right_tree}

    return build(X, y, 0)
    
def gini(y: np.ndarray):
    if not len(y):
        return 0
    else:
        _, counts = np.unique(y, return_counts=True)
        fract = counts/len(y)
        gini = 1 - np.sum(fract **2)
    return gini
    

def best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """Return the (feature_index, threshold) that maximazes gain."""
    
    best_feature = None
    best_threshold = None
    best_gain = 0
    n, d = X.shape

    parent_gini = gini(y)

    for j in range(d):
        col = X[:, j]
        thresholds = np.unique(col)
        midpoints = [(l[0] + l[1])/2 for l in zip(thresholds[:len(thresholds) - 1], thresholds[1:])]

        thresholds = midpoints

        for threshold in thresholds:
            mask = col <= threshold
            left = y[mask]
            right = y[~mask]
            gini_left = gini(left)
            gini_right = gini(right)
            weighted_score = len(left)/n * gini_left + len(right)/n * gini_right
            
            gain = parent_gini - weighted_score

            if best_gain < gain:
                best_feature = j
                best_threshold = threshold
                best_gain = gain

    return best_feature, best_threshold
    
