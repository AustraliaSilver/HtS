"""Device and accelerator detection utilities.

The package is PyTorch-first and can run on CPU, NVIDIA CUDA GPU, Apple MPS,
and TPU environments that provide `torch_xla`.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import random
from typing import Literal, Optional

import numpy as np
import torch

Backend = Literal["cpu", "cuda", "mps", "tpu"]


@dataclass(frozen=True)
class DeviceInfo:
    backend: Backend
    device: torch.device
    name: str
    available: bool = True
    notes: str = ""


def _try_tpu() -> Optional[DeviceInfo]:
    try:
        import torch_xla.core.xla_model as xm  # type: ignore

        dev = xm.xla_device()
        return DeviceInfo("tpu", dev, "TPU via torch_xla", True, "torch_xla detected")
    except Exception:
        return None


def detect_device(prefer: str = "auto") -> DeviceInfo:
    """Detect the best execution device.

    Parameters
    ----------
    prefer:
        One of: ``auto``, ``cpu``, ``cuda``, ``gpu``, ``mps``, ``tpu``.

    Notes
    -----
    TPU detection is attempted first in ``auto`` mode if `torch_xla` is
    available. CUDA is selected next, then Apple MPS, then CPU.
    """

    prefer = prefer.lower().strip()
    if prefer in {"cpu"}:
        return DeviceInfo("cpu", torch.device("cpu"), "CPU")

    if prefer in {"tpu", "xla"}:
        info = _try_tpu()
        if info is None:
            raise RuntimeError("TPU requested but torch_xla/XLA device is not available.")
        return info

    if prefer in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU requested but torch.cuda.is_available() is False.")
        idx = torch.cuda.current_device()
        return DeviceInfo("cuda", torch.device("cuda"), torch.cuda.get_device_name(idx))

    if prefer == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS requested but it is not available.")
        return DeviceInfo("mps", torch.device("mps"), "Apple Metal Performance Shaders")

    if prefer != "auto":
        raise ValueError(f"Unknown device preference: {prefer}")

    # Auto order: TPU -> CUDA -> MPS -> CPU.
    tpu = _try_tpu()
    if tpu is not None:
        return tpu
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        return DeviceInfo("cuda", torch.device("cuda"), torch.cuda.get_device_name(idx))
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return DeviceInfo("mps", torch.device("mps"), "Apple Metal Performance Shaders")
    return DeviceInfo("cpu", torch.device("cpu"), os.uname().machine if hasattr(os, "uname") else "CPU")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_xla_step(backend: str) -> None:
    """Call XLA optimizer step barrier when running on TPU."""
    if backend == "tpu":
        try:
            import torch_xla.core.xla_model as xm  # type: ignore

            xm.mark_step()
        except Exception:
            pass
