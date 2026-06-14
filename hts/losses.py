from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F


def cross_entropy_with_margin(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin_weight: float = 0.05,
    margin: float = 0.35,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross entropy plus a margin objective.

    The margin objective encourages the correct logit to be larger than the
    strongest wrong logit by at least ``margin``. This was introduced in B12 to
    convert better calibration/loss into higher discrete accuracy.
    """
    ce = F.cross_entropy(logits, labels)
    if margin_weight <= 0:
        return ce, ce, torch.zeros((), device=logits.device)
    correct = logits.gather(1, labels[:, None]).squeeze(1)
    wrong = logits.masked_fill(F.one_hot(labels, logits.size(-1)).bool(), -1e9).max(dim=1).values
    margin_loss = F.relu(margin - (correct - wrong)).mean()
    return ce + margin_weight * margin_loss, ce, margin_loss


def hts_regularization_loss(
    model,
    budget_weight: float = 1e-4,
    binary_weight: float = 1e-4,
    ratio_weight: float = 5e-4,
    task_offset_weight: float = 1e-5,
) -> Tuple[torch.Tensor, dict]:
    budget, binary, ratio, offset = model.hts_losses() if hasattr(model, "hts_losses") else (None, None, None, None)
    if budget is None:
        dev = next(model.parameters()).device
        budget = binary = ratio = offset = torch.tensor(0.0, device=dev)
    reg = budget_weight * budget + binary_weight * binary + ratio_weight * ratio + task_offset_weight * offset
    parts = {
        "reg_total": float(reg.detach()),
        "budget_loss": float(budget.detach()),
        "binary_loss": float(binary.detach()),
        "ratio_loss": float(ratio.detach()),
        "task_offset_l2": float(offset.detach()),
    }
    return reg, parts
