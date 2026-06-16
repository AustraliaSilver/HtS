"""Diagnostics and utility functions."""
from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().detach().cpu())


def average_dicts(items: Iterable[Dict[str, float]]) -> Dict[str, float]:
    items = list(items)
    if not items:
        return {}
    keys = sorted({k for d in items for k in d})
    out: Dict[str, float] = {}
    for k in keys:
        vals = [d[k] for d in items if k in d]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out
