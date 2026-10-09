import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    X = torch.tensor(features, dtype=torch.float32)   # (N, D)
    y = torch.tensor(labels, dtype=torch.float32)     # (N,)
    w = torch.tensor(weights, dtype=torch.float32)    # (D,)

    z = torch.matmul(X, w) + bias                     # (N, D) @ (D,) -> (N,)
    probs = torch.sigmoid(z)                          # (N,)

    mse = F.mse_loss(probs, y)

    probs_out = [round(p, 4) for p in probs.tolist()]
    return probs_out, round(mse.item(), 4)