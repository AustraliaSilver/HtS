"""Define and train a fully custom model group.

Run:
    python examples/custom_group.py
"""
from __future__ import annotations

import torch

from hts_b12 import (
    HtSBatch,
    HtSB12Classifier,
    LabelSpec,
    ModelGroupConfig,
    ModelGroupRegistry,
    TaskSpec,
    TrainConfig,
    build_hts_config_from_group,
)
from hts_b12.training import train_group_classifier


def make_parity_batch(batch_size: int, device: torch.device, seed: int) -> HtSBatch:
    g = torch.Generator(device="cpu").manual_seed(seed)
    max_len = 32
    lengths = torch.randint(4, max_len + 1, (batch_size,), generator=g)
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.long)
    task_ids = torch.zeros(batch_size, dtype=torch.long)

    for i, L in enumerate(lengths.tolist()):
        toks = torch.randint(1, 20, (L,), generator=g)
        input_ids[i, :L] = toks
        attention_mask[i, :L] = 1
        labels[i] = int(toks.sum().item() % 2)  # 0 even, 1 odd

    return HtSBatch(input_ids.to(device), task_ids.to(device), labels.to(device), attention_mask.to(device), group="toy_parity")


def main() -> None:
    registry = ModelGroupRegistry()
    group = ModelGroupConfig(
        name="toy_parity",
        vocab_size=32,
        max_length=32,
        tasks=[TaskSpec("sum_parity", 0, "Predict whether token sum is even or odd.")],
        labels=LabelSpec(num_classes=2, names=["even", "odd"]),
        recommended_model={"d_model": 64, "n_heads": 4, "num_layers": 1, "dim_ff": 128, "rank_main": 4, "rank_corr": 2},
    )
    registry.register(group, make_parity_batch)

    model = HtSB12Classifier(build_hts_config_from_group(group))
    log = train_group_classifier(model, "toy_parity", TrainConfig(steps=2, batch_size=4, eval_every=1), registry=registry)
    print(log.rows[-1])


if __name__ == "__main__":
    main()
