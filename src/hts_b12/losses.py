"""Loss functions for HtS-B12."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class MarginLoss(nn.Module):
    def __init__(self, margin: float = 0.6):
        super().__init__()
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        true = logits.gather(1, labels[:, None]).squeeze(1)
        wrong = logits.masked_fill(F.one_hot(labels, logits.size(-1)).bool(), -1e9).max(dim=1).values
        return F.relu(self.margin - (true - wrong)).mean()


@dataclass
class LossBreakdown:
    loss: torch.Tensor
    ce_loss: torch.Tensor
    margin_loss: torch.Tensor
    budget_reg: torch.Tensor
    binary_reg: torch.Tensor
    ratio_reg: torch.Tensor
    task_offset_reg: torch.Tensor

    def scalars(self) -> Dict[str, float]:
        return {k: float(v.detach().cpu()) for k, v in self.__dict__.items()}


class HtSB12Objective(nn.Module):
    """Cross-entropy + margin + HtS safety regularizers."""

    def __init__(
        self,
        margin: float = 0.6,
        margin_weight: float = 0.03,
        ratio_reg: float = 1e-3,
        budget_reg: float = 0.0,
        binary_reg: float = 0.0,
        task_offset_reg: float = 0.0,
        warmup_steps: int = 0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.margin = MarginLoss(margin)
        self.margin_weight = margin_weight
        self.ratio_reg = ratio_reg
        self.budget_reg = budget_reg
        self.binary_reg = binary_reg
        self.task_offset_reg = task_offset_reg
        self.warmup_steps = warmup_steps
        self.label_smoothing = label_smoothing

    def forward(self, model: nn.Module, logits: torch.Tensor, labels: torch.Tensor, step: int = 0) -> LossBreakdown:
        ce = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        ml = self.margin(logits, labels)
        warm = 1.0 if self.warmup_steps <= 0 else min(1.0, step / max(1, self.warmup_steps))
        dev = logits.device
        budget = binary = ratio = offset = torch.zeros((), device=dev)
        if hasattr(model, "hts_regularizers"):
            budget, binary, ratio, offset = model.hts_regularizers()
        breg = self.budget_reg * warm * budget
        bireg = self.binary_reg * warm * binary
        rreg = self.ratio_reg * warm * ratio
        oreg = self.task_offset_reg * warm * offset
        total = ce + self.margin_weight * warm * ml + breg + bireg + rreg + oreg
        return LossBreakdown(total, ce, ml, breg, bireg, rreg, oreg)
