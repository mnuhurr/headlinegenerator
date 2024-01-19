import math
import torch
import torch.nn.functional as F


def model_size(model: torch.nn.Module) -> int:
    num_params = sum(param.numel() for param in model.parameters())
    return num_params


def masked_loss(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(y_pred.permute(0, 2, 1), y_true, reduction='none')
    mask = (mask == 0).to(loss.dtype)
    return torch.sum(loss * mask) / torch.sum(mask).to(loss.dtype)


def step_lr(step: int, warmup_steps: int = 4000) -> float:
    # learning rate from the original attention paper modified in such a way that
    # this function peaks at 1, tune learning rate with optimizer
    arg1 = torch.tensor(1 / math.sqrt(step)) if step > 0 else torch.tensor(float('inf'))
    arg2 = torch.tensor(step * warmup_steps**-1.5)

    return math.sqrt(warmup_steps) * torch.minimum(arg1, arg2)
