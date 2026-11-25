import numpy as np


def purged_walk_forward_splits(n_samples: int, n_splits: int, embargo: int) -> list:
    """Build expanding-window train/test splits with an embargo gap.

    Args:
        n_samples (int): total number of chronologically ordered samples.
        n_splits (int): number of test folds.
        embargo (int): samples immediately before each test block to drop from training.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: one (train_idx, test_idx) pair per fold.
    """
    
    fold_size = n_samples//(n_splits + 1)
    folds = []

    for k in range(1, n_splits + 1):
        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k != n_splits else n_samples

        training_end = max(test_start - embargo, 0)
        test = np.arange(test_start, test_end)
        train = np.arange(training_end)

        folds.append((train, test))

    return folds

