import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).
    
    Args:
        y_true: List or array of true binary labels (0 or 1)
        y_prob: List or array of predicted probabilities for the positive class
        n_bins: Number of bins for grouping predictions (default: 10)
    
    Returns:
        float: ECE value rounded to 3 decimal places
    """

    scores = []
    y_prob_arr = np.array(y_prob)
    y_true_arr = np.array(y_true)

    for k in range(n_bins):
        lower = k/n_bins
        upper = (k + 1)/n_bins
        
        if k == 0:
           mask = (lower <= y_prob_arr) & (y_prob_arr<= upper)
        else:
            mask = (lower < y_prob_arr) & (y_prob_arr <= upper)

        if np.sum(mask) != 0:
            conf = np.mean(y_prob_arr[mask])
            acc = np.sum(y_true_arr[mask])/np.sum(mask)
            weight = np.sum(mask)/len(y_true_arr)

            scores.append(weight * np.abs(conf - acc))

    return float(round(np.sum(scores), 3))

