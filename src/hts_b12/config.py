"""Configuration objects for HtS-B12 models."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class HtSB12Config:
    """Configuration for :class:`HtSB12Classifier`.

    HtS-B12 keeps the original Hard-to-Soft philosophy:
    a hard controller generates soft, task-conditioned updates inside the true
    FFN computation path. The generated soft weights are trained end-to-end by
    task loss, not by direct supervision.
    """

    vocab_size: int = 128
    num_tasks: int = 8
    num_classes: int = 128
    max_length: int = 128

    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 2
    dim_ff: int = 256
    dropout: float = 0.1
    norm_first: bool = True

    # HtS generated-update capacity.
    task_dim: int = 32
    rank_main: int = 8
    rank_corr: int = 4

    # B12 routing/generation controls.
    alpha_max: float = 1.20
    target_min: float = 0.25
    target_max: float = 0.90
    corr_alpha_max: float = 0.55
    corr_gain: float = 6.0
    task_offset_scale: float = 0.30

    # Safety constraints.
    ratio_ceiling: float = 0.95
    corr_ceiling: float = 0.35

    # Pooling/classification.
    pool: str = "cls"  # "cls" or "mean"
    use_cls_token: bool = True

    # Loss defaults.
    margin: float = 0.60
    margin_weight: float = 0.03
    ratio_reg: float = 1e-3
    budget_reg: float = 0.0
    binary_reg: float = 0.0
    task_offset_reg: float = 0.0

    def __post_init__(self) -> None:
        positive_ints = {
            "vocab_size": self.vocab_size,
            "num_tasks": self.num_tasks,
            "num_classes": self.num_classes,
            "max_length": self.max_length,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "num_layers": self.num_layers,
            "dim_ff": self.dim_ff,
            "task_dim": self.task_dim,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.rank_main <= 0 or self.rank_corr <= 0:
            raise ValueError("rank_main and rank_corr must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.pool not in {"cls", "mean"}:
            raise ValueError("pool must be 'cls' or 'mean'")
        if self.target_min > self.target_max:
            raise ValueError("target_min must be <= target_max")
        if self.ratio_ceiling <= 0.0 or self.corr_ceiling <= 0.0:
            raise ValueError("ratio_ceiling and corr_ceiling must be > 0")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HtSB12Config":
        return cls(**data)

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "config.json").open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "HtSB12Config":
        with (Path(path) / "config.json").open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class TrainConfig:
    steps: int = 5000
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_steps: int = 250
    grad_clip: float = 1.0
    eval_every: int = 250
    device: str = "auto"
    seed: int = 42
    mixed_precision: bool = False
    save_best: bool = True
    output_dir: Optional[str] = None
