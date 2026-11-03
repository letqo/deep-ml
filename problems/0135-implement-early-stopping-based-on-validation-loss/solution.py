from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:

    wait = 0
    best = float('inf')
    best_epoch = 0

    for epoch, loss in enumerate(val_losses):
        
        if best - loss > min_delta:
            best = loss
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait == patience:
                return epoch, best_epoch
    
    return len(val_losses) - 1, best_epoch

