
import numpy as np

def gini_impurity(y):
    """
    Calculate Gini Impurity for a list of class labels.

    :param y: List of class labels
    :return: Gini Impurity rounded to three decimal places
    """
    y = np.array(y)
    _, counts = np.unique(y, return_counts=True)
    fract = np.array(counts)/len(y)
    gini = 1 - np.sum(fract **2)
    return round(gini,3)