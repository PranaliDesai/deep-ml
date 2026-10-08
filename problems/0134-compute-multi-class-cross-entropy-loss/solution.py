import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    p = np.clip(predicted_probs, epsilon, 1.0 - epsilon)
    return float(-np.mean(np.sum(true_labels * np.log(p), axis=1)))