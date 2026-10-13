import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor,
                 initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    X = torch.as_tensor(features, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.float64)

    w = torch.as_tensor(initial_weights, dtype=torch.float64).clone().requires_grad_(True)
    b = torch.tensor(float(initial_bias), dtype=torch.float64, requires_grad=True)

    mse_values = []
    for _ in range(epochs):
        preds = torch.sigmoid(X @ w + b)
        loss = torch.mean((preds - y) ** 2)
        mse_values.append(round(loss.item(), 4))

        loss.backward()
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()

    updated_weights = [round(v, 4) for v in w.detach().tolist()]
    updated_bias = round(b.item(), 4)
    return updated_weights, updated_bias, mse_values