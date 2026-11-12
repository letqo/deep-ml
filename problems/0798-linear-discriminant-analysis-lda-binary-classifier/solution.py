import numpy as np

def lda_predict(X_train, y_train, X_test):
    """
    Fit a binary LDA classifier on (X_train, y_train) and predict labels for X_test.

    Args:
        X_train: array-like of shape (n_samples, n_features)
        y_train: array-like of shape (n_samples,) with values in {0, 1}
        X_test:  array-like of shape (n_test, n_features)

    Returns:
        List[int] of predicted labels (0 or 1) for each row in X_test.
    """
    y_train = np.array(y_train)
    X_train, X_test = np.array(X_train), np.array(X_test)
    classes = np.unique(y_train)
    mean, m = [], []

    for idx, c in enumerate(classes):
        mean.append(np.mean(X_train[y_train == c], axis=0))
        D = X_train[y_train == c] - mean[idx]
        m.append(D.T@D)
    m = np.array(m)
    sw = np.sum(m, axis=0)
    w = np.linalg.solve(sw, mean[1] - mean[0])

    threshold = w @ (mean[0] + mean[1])/2

    y_pred = X_test@w > threshold

    return y_pred.astype('int').tolist()
