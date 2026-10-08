import torch

def to_float_tensor(values):
    return torch.as_tensor(values, dtype=torch.float32)
    
