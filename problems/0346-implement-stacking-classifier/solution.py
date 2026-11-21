import numpy as np

def stacking_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                        base_classifiers: list, meta_classifier, n_folds: int = 5) -> np.ndarray:
    """
    Implement a stacking classifier ensemble.
    
    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,)
        X_test: Test features of shape (m_samples, n_features)
        base_classifiers: List of classifier functions
        meta_classifier: Meta-level classifier function
        n_folds: Number of cross-validation folds
    
    Returns:
        np.ndarray: Final predictions on X_test
    """
    n, p = X_train.shape
    size = n // n_folds
    folds = []
    positions = np.arange(n)
    for k in range(n_folds):
        if k == n_folds - 1:
            folds.append([k * size, n])
        else:
            folds.append([k * size, (k + 1) * size])

    n_experts = len(base_classifiers)
    table = np.zeros((n, n_experts))
    for i in range(len(folds)):
        held_out_rows = folds[i]
        #Training rows mask
        mask = (positions >= held_out_rows[0]) & (positions < held_out_rows[-1])
        
        for j, clf in enumerate(base_classifiers):
            preds = clf(X_train[~mask], y_train[~mask], X_train[mask])
            table[mask, j] = preds

    table_manager = np.zeros((len(X_test), n_experts))
    for j, clf in enumerate(base_classifiers):
        preds = clf(X_train, y_train, X_test)
        table_manager[:, j] = preds

    trust = np.array(meta_classifier(table, y_train, table_manager)).astype(int)

    return trust
