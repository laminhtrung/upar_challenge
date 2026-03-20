import torch
import torch.nn as nn


def build_loss(loss_name="bce", pos_weight=None, weight=None, device=None):
    loss_name = loss_name.lower()

    if loss_name == "bce":
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight, dtype=torch.float32)

        if device is not None:
            if pos_weight is not None:
                pos_weight = pos_weight.to(device)
            if weight is not None:
                weight = weight.to(device)

        return nn.BCEWithLogitsLoss(
            weight=weight,
            pos_weight=pos_weight
        )

    raise ValueError(f"Unsupported loss: {loss_name}")