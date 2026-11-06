import numpy as np

def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC).
    
    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    P = np.sum(np.array(y_true) == 1)
    N = np.sum(np.array(y_true) == 0)

    thresholds = np.sort(np.array(list(set(y_scores)) + [float('inf')]))[:: -1]

    TP, FP = 0, 0
    tpr_list, fpr_list, to_sum = [], [], []

    for threshold in thresholds:
        for y_score, y in zip(y_scores, y_true):
            if y_score >= threshold and y == 1:
                TP += 1
            if y_score >= threshold and y == 0:
                FP += 1
        tpr_list.append(TP/P if P!=0 else 0.0)
        fpr_list.append(FP/N if N!=0 else 0.0)
        TP, FP = 0, 0
    for tpr_prev, tpr_next, fpr_prev, fpr_next in zip(tpr_list[:-1], tpr_list[1:], fpr_list[:-1], fpr_list[1:]):
       to_sum.append((tpr_prev + tpr_next)/2 * (fpr_next - fpr_prev))
    
    return np.sum(to_sum)


    