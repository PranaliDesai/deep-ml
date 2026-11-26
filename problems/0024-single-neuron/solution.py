import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    X = np.array(features)
    y = np.array(labels)
    w = np.array(weights)

    # Weighted sum
    z = np.dot(X, w) + bias

    # Sigmoid activation
    predictions = 1 / (1 + np.exp(-z))

    # Mean squared error
    loss = np.mean((predictions - y) ** 2)

    return predictions.tolist(), float(loss)
