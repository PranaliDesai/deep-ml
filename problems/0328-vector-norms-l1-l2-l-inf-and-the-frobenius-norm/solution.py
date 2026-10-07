import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.
    'l1', 'l2', and 'linf' are entrywise norms and accept a 1D or 2D array.
    'frobenius' is a matrix norm and raises a ValueError if arr is not 2D.
    """
    # Validate dimensions for matrix norms
    if norm_type == "frobenius" and arr.ndim != 2:
        raise ValueError(f"Expected a 2D array for frobenius norm, but got {arr.ndim}D")
        
    # Compute norms
    if norm_type == "l1":
        # Entrywise L1: Sum of absolute values of all elements
        return float(np.sum(np.abs(arr)))
        
    elif norm_type == "l2":
        # Entrywise L2: Square root of the sum of squared values
        return float(np.sqrt(np.sum(np.square(arr))))
        
    elif norm_type == "linf":
        # Entrywise L-infinity: Maximum absolute value
        return float(np.max(np.abs(arr)))
        
    elif norm_type == "frobenius":
        # Frobenius norm for 2D matrices
        return float(np.linalg.norm(arr, ord="fro"))
        
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")
