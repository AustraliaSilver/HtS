"""Built-in model-group presets."""
from __future__ import annotations

import torch

from .groups import HtSBatch, LabelSpec, ModelGroupConfig, TaskSpec, GLOBAL_REGISTRY
from .data.string_tasks import make_string_count_batch


def string_length_count_group(max_length: int = 64, vocab_size: int = 128, num_classes: int = 128) -> ModelGroupConfig:
    return ModelGroupConfig(
        name="string_length_count",
        vocab_size=vocab_size,
        max_length=max_length,
        tasks=[
            TaskSpec("length", 0, "Predict sequence length."),
            TaskSpec("count_a", 1, "Count occurrences of token 'a'."),
            TaskSpec("count_b", 2, "Count occurrences of token 'b'."),
            TaskSpec("count_digit", 3, "Count digit-like tokens."),
            TaskSpec("count_upper", 4, "Count upper-case-like tokens."),
            TaskSpec("parity", 5, "Predict length/count parity."),
            TaskSpec("first_last_match", 6, "Classify whether first and last symbols match."),
            TaskSpec("bucket_length", 7, "Predict coarse length bucket."),
        ],
        labels=LabelSpec(num_classes=num_classes),
        description="Built-in synthetic string length/count benchmark group.",
        recommended_model={
            "d_model": 128,
            "n_heads": 4,
            "num_layers": 2,
            "dim_ff": 256,
            "rank_main": 8,
            "rank_corr": 4,
            "task_dim": 32,
        },
        recommended_training={
            "lr": 1e-3,
            "warmup_steps": 250,
            "grad_clip": 1.0,
            "weight_decay": 0.01,
        },
    )


def register_builtin_groups() -> None:
    group = string_length_count_group()

    def factory(batch_size: int, device: torch.device, seed: int) -> HtSBatch:
        b = make_string_count_batch(batch_size=batch_size, max_length=group.max_length, device=device, num_classes=group.num_classes, seed=seed)
        return HtSBatch(b.input_ids, b.task_ids, b.labels, b.attention_mask, group=group.name)

    GLOBAL_REGISTRY.register(group, factory)


register_builtin_groups()
