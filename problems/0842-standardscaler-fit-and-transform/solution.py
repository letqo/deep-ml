import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Fit a standard scaler on X_train and transform X_test.
    Returns the standardized X_test as a numpy array.
    """
    mean = np.mean(X_train, axis=0)
    
    variance = np.mean((X_train - mean)** 2, axis=0)
    std = np.sqrt(variance)
    std = np.where(std == 0, 1.0, std)

    standardized_X_train = (X_train - mean)/std
    standardized_X_test = (X_test - mean)/std
    
    return standardized_X_test
