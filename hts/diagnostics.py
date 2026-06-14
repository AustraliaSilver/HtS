from __future__ import annotations
from typing import Dict
import torch


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().detach())


def collect_diagnostics(model) -> Dict[str, float]:
    if hasattr(model, "diagnostics"):
        return model.diagnostics()
    return {}


def summarize_batch(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = logits.softmax(dim=-1)
    top = probs.max(dim=-1).values
    return {
        "accuracy": accuracy(logits, labels),
        "mean_confidence": float(top.mean().detach()),
        "mean_true_prob": float(probs.gather(1, labels[:, None]).mean().detach()),
    }
