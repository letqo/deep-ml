import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
    """
    Compute entropy of P and cross-entropy between P and Q.
    
    Args:
        P: True probability distribution
        Q: Predicted probability distribution
    
    Returns:
        Tuple of (entropy H(P), cross-entropy H(P,Q))
    """
    h_p, h_pq = 0, 0
    for p, q in zip(P, Q):
        if p == 0:
            continue
        else:
            h_p += - p * np.log(p)
        if q == 0:
            h_pq = float('inf')
        else:
            h_pq += - p * np.log(q)

    return h_p, h_pq
