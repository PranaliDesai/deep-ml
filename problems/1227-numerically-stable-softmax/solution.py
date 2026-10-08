import torch

def softmax(t: torch.Tensor, dim: int) -> torch.Tensor:
    t_max = t.max(dim=dim, keepdim=True).values
    e = torch.exp(t - t_max)
    return e / e.sum(dim=dim, keepdim=True)
