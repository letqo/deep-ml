import numpy as np

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class
    
    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    P = np.sum(np.array(y_true) == 1)
    N = np.sum(np.array(y_true) == 0)

    thresholds = np.sort(np.array(list(set(y_scores)) + [float('inf')]))[:: -1]

    TP, FP = 0, 0
    TPR_list, FPR_list = [], []

    for threshold in thresholds:
        for y_score, y in zip(y_scores, y_true):
            if y_score >= threshold and y == 1:
                TP += 1
            if y_score >= threshold and y == 0:
                FP += 1
        TPR_list.append(TP/P if P!=0 else 0.0)
        FPR_list.append(FP/N if N!=0 else 0.0)
        TP, FP = 0, 0

    return FPR_list, TPR_list

    
