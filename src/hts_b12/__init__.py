"""HtS-B12: Hard-to-Soft generated-computation library.

Public API mirrors the style of common Transformer libraries while exposing
HtS-specific diagnostics, model-group configuration, and task adapters.
"""
from .config import HtSB12Config, TrainConfig
from .device import DeviceInfo, detect_device, seed_everything
from .diagnostics import accuracy, count_parameters
from .groups import (
    GLOBAL_REGISTRY,
    HtSBatch,
    LabelSpec,
    ModelGroupConfig,
    ModelGroupRegistry,
    TaskSpec,
    build_hts_config_from_group,
    make_multi_group_factory,
)
from .losses import HtSB12Objective, MarginLoss
from .modeling import HtSB12Classifier, TransformerClassifier
from .presets import register_builtin_groups, string_length_count_group

__all__ = [
    "HtSB12Config",
    "TrainConfig",
    "DeviceInfo",
    "detect_device",
    "seed_everything",
    "accuracy",
    "count_parameters",
    "HtSB12Objective",
    "MarginLoss",
    "HtSB12Classifier",
    "TransformerClassifier",
    "HtSBatch",
    "TaskSpec",
    "LabelSpec",
    "ModelGroupConfig",
    "ModelGroupRegistry",
    "GLOBAL_REGISTRY",
    "build_hts_config_from_group",
    "make_multi_group_factory",
    "register_builtin_groups",
    "string_length_count_group",
]
