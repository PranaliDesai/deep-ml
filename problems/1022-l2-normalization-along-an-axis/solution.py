import numpy as np

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> list:
    """
    L2-normalize x along the given axis.

    Args:
        x: input NumPy array
        axis: axis along which to normalize
        eps: small constant for numerical stability

    Returns:
        Normalized array as a nested Python list.
    """
    # 1. Compute the L2 norm (Euclidean length) along the specified axis
    #    keepdims=True maintains the dimensions so we can divide correctly later
    norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    
    # 2. Add eps to the norm to prevent division by zero errors
    norm = np.maximum(norm, eps)
    
    # 3. Divide vectorized elements by the norm and convert the final result to a Python list
    normalized_array = x / norm
    return normalized_array.tolist()
