"""Synthetic string length/count benchmark used for quick HtS validation.

The generator returns task-conditioned sequence classification batches. It is
intentionally lightweight so users can test CPU/GPU/TPU plumbing before scaling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

PAD = 0
TOKENS = {
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "0": 5,
    "1": 6,
    "2": 7,
    "3": 8,
    "x": 9,
    "y": 10,
}

TASKS: Dict[str, int] = {
    "length": 0,
    "count_a": 1,
    "count_b": 2,
    "count_digit": 3,
    "count_vowel_like": 4,
}


@dataclass
class StringBatch:
    input_ids: torch.Tensor
    task_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor


def make_string_count_batch(
    batch_size: int,
    max_length: int,
    device: torch.device | str = "cpu",
    min_length: int = 1,
    num_classes: int = 128,
    seed: int | None = None,
    task_mix: Tuple[str, ...] = ("length", "count_a", "count_b", "count_digit"),
) -> StringBatch:
    """Generate a batch for length/count tasks.

    Labels are integer counts clipped to ``num_classes - 1``.
    """
    dev = torch.device(device)
    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
    else:
        gen = None

    ids = torch.zeros(batch_size, max_length, dtype=torch.long)
    mask = torch.zeros(batch_size, max_length, dtype=torch.long)
    tasks = torch.empty(batch_size, dtype=torch.long)
    labels = torch.empty(batch_size, dtype=torch.long)

    vocab_vals = torch.tensor(list(TOKENS.values()), dtype=torch.long)
    digit_vals = torch.tensor([TOKENS["0"], TOKENS["1"], TOKENS["2"], TOKENS["3"]], dtype=torch.long)
    vowel_like_vals = torch.tensor([TOKENS["a"]], dtype=torch.long)

    for i in range(batch_size):
        length = int(torch.randint(min_length, max_length + 1, (1,), generator=gen).item())
        seq = vocab_vals[torch.randint(0, len(vocab_vals), (length,), generator=gen)]
        ids[i, :length] = seq
        mask[i, :length] = 1
        task_name = task_mix[int(torch.randint(0, len(task_mix), (1,), generator=gen).item())]
        task_id = TASKS[task_name]
        tasks[i] = task_id
        if task_name == "length":
            y = length
        elif task_name == "count_a":
            y = int((seq == TOKENS["a"]).sum().item())
        elif task_name == "count_b":
            y = int((seq == TOKENS["b"]).sum().item())
        elif task_name == "count_digit":
            y = int(torch.isin(seq, digit_vals).sum().item())
        elif task_name == "count_vowel_like":
            y = int(torch.isin(seq, vowel_like_vals).sum().item())
        else:
            raise KeyError(task_name)
        labels[i] = min(y, num_classes - 1)

    return StringBatch(ids.to(dev), tasks.to(dev), labels.to(dev), mask.to(dev))
