import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    indices = [i for i in range(n_samples)]
    if shuffle:
        np.random.shuffle(indices)

    fold_size = n_samples//k
    remainder = n_samples%k

    folds_indices = []
    pieces = []
    s = 0

    for fold in range(k):
        if fold < remainder:
            size = fold_size + 1
        else:
            size = fold_size
        pieces.append(indices[s : s + size])
        s = s + size

    for fold in range(k):
        concat = []

        for i in range(fold):
            concat += pieces[i]
        for j in range(fold + 1, k):
            concat += pieces[j]
        folds_indices.append((concat, pieces[fold]))
        
    return folds_indices

