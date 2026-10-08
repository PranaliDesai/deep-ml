import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
    entropy = -sum(p * np.log(p) for p in P if p > 0)
    cross_entropy = -sum(p * np.log(q) for p, q in zip(P, Q) if p > 0)
    return float(entropy), float(cross_entropy)