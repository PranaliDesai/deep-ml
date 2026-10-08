import math

def activation_derivatives(x: float) -> dict[str, float]:
    sig = 1 / (1 + math.exp(-x))
    t = math.tanh(x)
    return {
        'sigmoid': sig * (1 - sig),
        'tanh': 1 - t ** 2,
        'relu': 1.0 if x > 0 else 0.0,
    }