import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    mean = np.mean(X, axis=-1, keepdims=True)               # per-sample mean over features
    var = np.var(X, axis=-1, keepdims=True)                 # per-sample variance (population)
    X_norm = (X - mean) / np.sqrt(var + epsilon)            # normalize
    return gamma * X_norm + beta                            # scale and shift