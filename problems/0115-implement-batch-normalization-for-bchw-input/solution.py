import numpy as np


def batch_normalization(X, gamma, beta, running_mean=None, running_var=None,
                        momentum=0.1, epsilon=1e-5, training=True):
    C = X.shape[1]
    if running_mean is None:
        running_mean = np.zeros((1, C, 1, 1))
    if running_var is None:
        running_var = np.ones((1, C, 1, 1))

    if training:
        mean = np.mean(X, axis=(0, 2, 3), keepdims=True)   # per-channel
        var = np.var(X, axis=(0, 2, 3), keepdims=True)     # biased, used to normalize
        n = X.shape[0] * X.shape[2] * X.shape[3]
        running_mean = (1 - momentum) * running_mean + momentum * mean
        running_var = (1 - momentum) * running_var + momentum * var    
    else:
        mean, var = running_mean, running_var

    X_norm = (X - mean) / np.sqrt(var + epsilon)
    return gamma * X_norm + beta, running_mean, running_var