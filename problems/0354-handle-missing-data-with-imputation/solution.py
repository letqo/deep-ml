import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.
    
    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'
        
    Returns:
        2D numpy array with missing values imputed
    """

    data_copy = data.copy()
    nan_mask = np.isnan(data_copy)

    for i in range(len(data[0])):
        col = data_copy[:, i]
        dt = col[~nan_mask[:, i]]
            
        if strategy =='mean':
            filling = np.mean(dt)
        elif strategy == 'median':
            filling = np.median(dt)
        elif strategy == 'mode':
            values, counts = np.unique(dt, return_counts=True)
            filling = values[np.argmax(counts)]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        col[nan_mask[: ,i]] = filling 

    return data_copy


