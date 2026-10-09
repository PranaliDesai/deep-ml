import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    X = features.float()                                      # (N, D)
    y = labels.float()                                        # (N,)

    # Fresh leaf tensors that autograd will track
    w = initial_weights.clone().detach().float().requires_grad_(True)   # (D,)
    b = torch.tensor(float(initial_bias), requires_grad=True)           # scalar

    opt = torch.optim.SGD([w, b], lr=learning_rate)
    loss_fn = nn.MSELoss()
    mse_values = []

    for _ in range(epochs):
        preds = torch.sigmoid(X @ w + b)    # forward
        loss = loss_fn(preds, y)
        mse_values.append(round(loss.item(), 4))

        opt.zero_grad()                     # clear the old gradients
        loss.backward()                     # compute dL/dw and dL/db
        opt.step()                          # w -= lr * w.grad, b -= lr * b.grad

    updated_weights = [round(v, 4) for v in w.detach().tolist()]
    updated_bias = round(b.item(), 4)
    return updated_weights, updated_bias, mse_values