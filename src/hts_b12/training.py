"""Minimal training utilities for HtS-B12."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch import nn

from .config import TrainConfig
from .device import detect_device, maybe_xla_step, seed_everything
from .diagnostics import accuracy, count_parameters
from .losses import HtSB12Objective


@dataclass
class TrainLog:
    rows: List[Dict[str, float | int | str]]
    best_acc: float
    best_step: int


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)).item())


def train_classifier(
    model: nn.Module,
    batch_fn: Callable[[int, torch.device, int], object],
    config: TrainConfig,
    objective: Optional[HtSB12Objective] = None,
) -> TrainLog:
    """Train a classifier using generated batches.

    ``batch_fn`` must accept ``(batch_size, device, seed)`` and return an object
    with ``input_ids``, ``task_ids``, ``labels`` and optional ``attention_mask``.
    """
    seed_everything(config.seed)
    info = detect_device(config.device)
    device = info.device
    model.to(device)
    model.train()
    objective = objective or HtSB12Objective(warmup_steps=config.warmup_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rows: List[Dict[str, float | int | str]] = []
    best_acc = -1.0
    best_step = 0
    outdir = Path(config.output_dir) if config.output_dir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    for step in range(1, config.steps + 1):
        lr = cosine_with_warmup(step, config.steps, config.warmup_steps, config.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = batch_fn(config.batch_size, device, config.seed * 1_000_000 + step)
        logits = model(batch.input_ids, batch.task_ids, getattr(batch, "attention_mask", None))
        loss_bd = objective(model, logits, batch.labels, step=step)
        optimizer.zero_grad(set_to_none=True)
        loss_bd.loss.backward()
        if config.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        maybe_xla_step(info.backend)

        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            with torch.no_grad():
                acc = accuracy(logits, batch.labels)
            row: Dict[str, float | int | str] = {
                "step": step,
                "backend": info.backend,
                "lr": lr,
                "params": count_parameters(model),
                "accuracy": acc,
                **loss_bd.scalars(),
            }
            if hasattr(model, "hts_diagnostics"):
                row.update(model.hts_diagnostics())  # type: ignore[arg-type]
            rows.append(row)
            if acc > best_acc:
                best_acc = acc
                best_step = step
                if config.save_best and outdir and hasattr(model, "save_pretrained"):
                    model.save_pretrained(outdir / "best")
    return TrainLog(rows=rows, best_acc=best_acc, best_step=best_step)


def train_group_classifier(
    model: nn.Module,
    group_name: str,
    config: TrainConfig,
    objective: Optional[HtSB12Objective] = None,
    registry=None,
) -> TrainLog:
    """Train on a registered model group by name.

    This is the direct API most users need:

    ```python
    train_group_classifier(model, "string_length_count", TrainConfig(...))
    ```

    For custom projects, register a `ModelGroupConfig` and a batch factory in a
    `ModelGroupRegistry`, then pass that registry here.
    """

    from .groups import GLOBAL_REGISTRY

    registry = registry or GLOBAL_REGISTRY
    return train_classifier(model, registry.factory(group_name), config, objective=objective)
