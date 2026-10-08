import torch
import torch.nn as nn

def two_layer_mlp_forward(x, w1, b1, w2, b2) -> float:
    model = nn.Sequential(
        nn.Linear(2, 2),
        nn.ReLU(),
        nn.Linear(2, 1),
    )

    with torch.no_grad():
        model[0].weight.copy_(w1)   # (2, 2)
        model[0].bias.copy_(b1)     # (2,)
        model[2].weight.copy_(w2)   # (1, 2)
        model[2].bias.copy_(b2)     # (1,)

    return model(x).item()          # (1, 2) -> (1, 2) -> (1, 2) -> (1, 1)