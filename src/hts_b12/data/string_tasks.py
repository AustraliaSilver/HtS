"""Synthetic string length/count benchmark used for HtS validation.

The original generator used a Python loop over the batch.  That is fine for
small smoke tests but becomes a bottleneck on Kaggle/GPU.  This version keeps
exactly the same task semantics while generating the batch vectorially.
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
    Generation is deterministic for a given seed and vectorized on CPU before
    moving tensors to the requested device.  Keeping RNG on CPU makes results
    reproducible across CPU/GPU/TPU as much as PyTorch allows.
    """
    if min_length < 1:
        raise ValueError("min_length must be >= 1")
    if max_length < min_length:
        raise ValueError("max_length must be >= min_length")
    for task in task_mix:
        if task not in TASKS:
            raise KeyError(task)

    dev = torch.device(device)
    gen = torch.Generator(device="cpu") if seed is not None else None
    if gen is not None:
        gen.manual_seed(int(seed))

    lengths = torch.randint(min_length, max_length + 1, (batch_size,), generator=gen, dtype=torch.long)
    ids = torch.randint(1, len(TOKENS) + 1, (batch_size, max_length), generator=gen, dtype=torch.long)
    positions = torch.arange(max_length, dtype=torch.long).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)
    ids = ids.masked_fill(~mask, PAD)

    task_name_to_id = torch.tensor([TASKS[t] for t in task_mix], dtype=torch.long)
    sampled = torch.randint(0, len(task_mix), (batch_size,), generator=gen, dtype=torch.long)
    tasks = task_name_to_id[sampled]

    labels_by_task = {
        TASKS["length"]: lengths,
        TASKS["count_a"]: ((ids == TOKENS["a"]) & mask).sum(dim=1),
        TASKS["count_b"]: ((ids == TOKENS["b"]) & mask).sum(dim=1),
        TASKS["count_digit"]: (((ids >= TOKENS["0"]) & (ids <= TOKENS["3"])) & mask).sum(dim=1),
        TASKS["count_vowel_like"]: ((ids == TOKENS["a"]) & mask).sum(dim=1),
    }

    labels = torch.zeros(batch_size, dtype=torch.long)
    for task_id, values in labels_by_task.items():
        labels = torch.where(tasks == task_id, values, labels)
    labels = labels.clamp_max(num_classes - 1)

    return StringBatch(
        input_ids=ids.to(dev),
        task_ids=tasks.to(dev),
        labels=labels.to(dev),
        attention_mask=mask.to(dtype=torch.long).to(dev),
    )
