from __future__ import annotations
from dataclasses import dataclass


@dataclass
class HtSConfig:
    """Configuration for the B12-style HtS generated-computation block."""
    vocab_size: int = 64
    output_dim: int = 128
    num_tasks: int = 30
    max_len: int = 12
    d_model: int = 40
    n_heads: int = 4
    dim_ff: int = 64
    n_layers: int = 1
    task_dim: int = 16
    rank_main: int = 5
    rank_corr: int = 2
    alpha_max: float = 1.18
    target_min: float = 0.34
    target_max: float = 0.90
    tune_scale: float = 0.34
    gate_bias: float = -0.06
    task_offset_scale: float = 0.30
    corr_alpha_max: float = 0.55
    corr_gain: float = 0.55
    ratio_ceiling: float = 1.35
    corr_ceiling: float = 0.55
    correction_mode: str = "input"  # input, output, both, none
    dropout: float = 0.0


@dataclass
class TransformerConfig:
    """Configuration for a static Transformer baseline."""
    vocab_size: int = 64
    output_dim: int = 128
    num_tasks: int = 30
    max_len: int = 12
    d_model: int = 40
    n_heads: int = 4
    dim_ff: int = 64
    n_layers: int = 1
    dropout: float = 0.0


@dataclass
class TrainConfig:
    steps: int = 300
    batch_size: int = 64
    lr: float = 3e-3
    eval_every: int = 50
    eval_batches: int = 10
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda, mps, tpu
    margin_weight: float = 0.05
    margin: float = 0.35
    budget_weight: float = 1e-4
    binary_weight: float = 1e-4
    ratio_weight: float = 5e-4
    task_offset_weight: float = 1e-5
    grad_clip: float = 1.0
    benchmark: str = "synthetic"  # synthetic, multi_step, compositional
