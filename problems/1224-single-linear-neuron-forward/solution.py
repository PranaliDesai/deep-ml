import torch
import torch.nn as nn

def single_neuron_forward(x: torch.Tensor) -> float:
    neuron = nn.Linear(3, 1)

    with torch.no_grad():
        neuron.weight.copy_(torch.tensor([[0.5, -0.2, 0.3]]))  # shape (1, 3)
        neuron.bias.copy_(torch.tensor([0.1]))                 # shape (1,)

    out = neuron(x)          # (1, 3) @ (3, 1) + (1,) -> (1, 1)
    return out.item()
