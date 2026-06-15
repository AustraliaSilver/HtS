"""Model-group abstractions for HtS-B12.

This module makes HtS usable beyond the built-in string/count benchmark.  A
`ModelGroup` describes a family of tasks/classes/data adapters.  Users can
register many groups in one project and train/evaluate them through the same
HtS-B12 backbone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional
import json

try:  # optional dependency; the library works without PyYAML
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

import torch


@dataclass
class LabelSpec:
    """Class/label definition for one group.

    Parameters
    ----------
    num_classes:
        Number of output classes.  Keep all labels in `[0, num_classes)`.
    names:
        Optional human-readable labels.  Useful for reports and decoding.
    ignore_index:
        Optional ignore index for token-level tasks.
    """

    num_classes: int
    names: Optional[List[str]] = None
    ignore_index: int = -100


@dataclass
class TaskSpec:
    """A task inside a model group.

    A group may contain many tasks, e.g. `length`, `count_a`, `count_digit`,
    `parity`, or domain-specific tasks.  Each task receives one stable integer
    id used by HtS task routers.
    """

    name: str
    id: int
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelGroupConfig:
    """User-facing configuration for a model/task group.

    Examples
    --------
    ```python
    ModelGroupConfig(
        name="string_count",
        vocab_size=128,
        max_length=64,
        tasks=[TaskSpec("length", 0), TaskSpec("count_a", 1)],
        labels=LabelSpec(num_classes=128),
        recommended_model={"d_model": 128, "num_layers": 2},
    )
    ```
    """

    name: str
    vocab_size: int
    max_length: int
    tasks: List[TaskSpec]
    labels: LabelSpec
    pad_token_id: int = 0
    cls_token_id: Optional[int] = None
    description: str = ""
    recommended_model: Dict[str, Any] = field(default_factory=dict)
    recommended_training: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_tasks(self) -> int:
        return max((t.id for t in self.tasks), default=-1) + 1

    @property
    def num_classes(self) -> int:
        return self.labels.num_classes

    def task_id(self, name: str) -> int:
        for t in self.tasks:
            if t.name == name:
                return t.id
        raise KeyError(f"Unknown task {name!r} in group {self.name!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelGroupConfig":
        labels = data.get("labels", {})
        tasks = data.get("tasks", [])
        return cls(
            name=str(data["name"]),
            vocab_size=int(data["vocab_size"]),
            max_length=int(data["max_length"]),
            tasks=[TaskSpec(**dict(t)) for t in tasks],
            labels=LabelSpec(**dict(labels)),
            pad_token_id=int(data.get("pad_token_id", 0)),
            cls_token_id=data.get("cls_token_id"),
            description=str(data.get("description", "")),
            recommended_model=dict(data.get("recommended_model", {})),
            recommended_training=dict(data.get("recommended_training", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        if path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is not installed. Use .json or install pyyaml.")
            path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        else:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ModelGroupConfig":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is not installed. Use .json or install pyyaml.")
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)


@dataclass
class HtSBatch:
    """Standard batch object consumed by HtS models/trainers."""

    input_ids: torch.Tensor
    task_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    group: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "HtSBatch":
        return HtSBatch(
            input_ids=self.input_ids.to(device),
            task_ids=self.task_ids.to(device),
            labels=self.labels.to(device),
            attention_mask=None if self.attention_mask is None else self.attention_mask.to(device),
            group=self.group,
            metadata=self.metadata,
        )


BatchFactory = Callable[[int, torch.device, int], HtSBatch]


class ModelGroupRegistry:
    """Simple in-process registry for groups and their data factories."""

    def __init__(self) -> None:
        self._groups: Dict[str, ModelGroupConfig] = {}
        self._factories: Dict[str, BatchFactory] = {}

    def register(self, group: ModelGroupConfig, batch_factory: Optional[BatchFactory] = None) -> None:
        self._groups[group.name] = group
        if batch_factory is not None:
            self._factories[group.name] = batch_factory

    def get(self, name: str) -> ModelGroupConfig:
        return self._groups[name]

    def factory(self, name: str) -> BatchFactory:
        if name not in self._factories:
            raise KeyError(f"No batch factory registered for group {name!r}")
        return self._factories[name]

    def names(self) -> List[str]:
        return sorted(self._groups)

    def __contains__(self, name: str) -> bool:
        return name in self._groups


GLOBAL_REGISTRY = ModelGroupRegistry()


def build_hts_config_from_group(group: ModelGroupConfig, **overrides: Any):
    """Create `HtSB12Config` from a group with optional overrides."""

    from .config import HtSB12Config

    data = {
        "vocab_size": group.vocab_size,
        "num_tasks": group.num_tasks,
        "num_classes": group.num_classes,
        "max_length": group.max_length,
    }
    data.update(group.recommended_model)
    data.update(overrides)
    return HtSB12Config(**data)


def make_multi_group_factory(registry: ModelGroupRegistry, group_names: Iterable[str]) -> BatchFactory:
    """Build a mixed sampler over several registered groups.

    The returned factory uniformly samples one group per batch.  Use this for
    multi-domain experiments while preserving a direct, simple training API.
    """

    names = list(group_names)
    if not names:
        raise ValueError("group_names must not be empty")

    def factory(batch_size: int, device: torch.device, seed: int) -> HtSBatch:
        idx = int(torch.Generator().manual_seed(seed).initial_seed() % len(names))
        name = names[idx]
        return registry.factory(name)(batch_size, device, seed)

    return factory
