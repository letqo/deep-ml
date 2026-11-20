import numpy as np

def gini(y: np.ndarray):
    if not len(y):
        return 0
    else:
        _, counts = np.unique(y, return_counts=True)
        fract = counts/len(y)
        gini = 1 - np.sum(fract **2)
    return gini
    

def random_forest_fit_predict(X_train, y_train, X_test, n_trees: int = 5, max_depth: int = 3, max_features: int = 2, seed: int = 0):
    """
    Train a random forest classifier and predict on X_test.
    Returns a list of integer class predictions.
    """
    X_train, y_train, X_test = np.array(X_train), np.array(y_train), np.array(X_test)
    rng = np.random.RandomState(seed)
    N, D = X_train.shape

    def build(X, y, depth):
        leaf = {'leaf': True, 'prediction': int(np.bincount(y).argmax())}

        # (a) stopping conditions
        if len(np.unique(y)) == 1 or len(y) < 2 or depth >= max_depth:
            return leaf

        # (b) the random features for this node
        features = rng.choice(D, size=min(max_features, D), replace=False)

        # (c) search among those features only
        best_score = float('inf')
        best_feature, best_threshold = None, None

        for j in features:
            # midpoints, mask, weighted Gini, keep the lowest
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
                weighted_score = len(left)/len(y) * gini_left + len(right)/len(y) * gini_right

                if best_score > weighted_score:
                    best_feature = j
                    best_threshold = threshold
                    best_score = weighted_score

        if best_feature is None:
            return leaf

        # (d) split and recurse: left first, then right
        mask = X[:, best_feature] <= best_threshold
        left_tree = build(X[mask], y[mask], depth + 1)
        right_tree = build(X[~mask], y[~mask], depth + 1)

        return {'leaf': False, 'feature': int(best_feature), 'threshold': float(best_threshold),
                'left': left_tree, 'right': right_tree}

    def predict_tree(node, x):
    # Part 3: one tree, one point
        while node['leaf'] is not True:
            if x[node['feature']]  <= node['threshold']:
                node = node['left']
            else:
                node = node['right']   
        return node['prediction']

    trees = []
    for _ in range(n_trees):
        idx = rng.choice(N, size=N, replace=True)
        trees.append(build(X_train[idx], y_train[idx], 0))

    predictions = []

    for x in X_test:
        votes = [predict_tree(tree,x) for tree in trees]                 # each tree's answer for x
        predictions.append(int(np.bincount(votes).argmax()))       # majority, ties to the smallest label
    return predictions
