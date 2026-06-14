"""Hard-to-Soft (HtS) generated-computation architecture library.

This package contains a clean implementation of the final CPU-scale HtS line:
B12-style task-adaptive true-FFN soft-weight generation with ratio control,
correction gates, margin-oriented training, and CPU/GPU/TPU device detection.
"""
from .config import HtSConfig, TransformerConfig, TrainConfig
from .device import DeviceInfo, HtSDeviceManager, detect_device
from .layers import GeneratedDiagonalLinear, HtSB12FFN
from .models import HtSTransformerClassifier, StaticTransformerClassifier
from .losses import cross_entropy_with_margin, hts_regularization_loss

__version__ = "0.1.0"
__author__ = "Hung Anh Le"
__email__ = "hunganhl642@gmail.com"
__license__ = "MIT"

__all__ = [
    "HtSConfig", "TransformerConfig", "TrainConfig",
    "DeviceInfo", "HtSDeviceManager", "detect_device",
    "GeneratedDiagonalLinear", "HtSB12FFN",
    "HtSTransformerClassifier", "StaticTransformerClassifier",
    "cross_entropy_with_margin", "hts_regularization_loss",
    "__version__",
]
