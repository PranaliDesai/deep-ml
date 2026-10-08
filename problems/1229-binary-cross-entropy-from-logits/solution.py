import torch

def bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    x, y = logits, targets.float()
    loss = x.clamp(min=0) - x * y + torch.log(1 + torch.exp(-x.abs()))
    return round(loss.mean().item(), 4)