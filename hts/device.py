from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Any
import torch


@dataclass
class DeviceInfo:
    backend: str
    device: Any
    name: str
    is_tpu: bool = False
    is_cuda: bool = False
    is_mps: bool = False
    is_cpu: bool = False
    xla_model: Optional[Any] = None

    def __str__(self) -> str:
        return f"DeviceInfo(backend={self.backend}, device={self.device}, name={self.name})"


def _try_tpu() -> Optional[DeviceInfo]:
    """Try to detect PyTorch/XLA TPU without making torch-xla a hard dependency."""
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        # In Kaggle/Colab TPU, this succeeds only when XLA runtime is available.
        dev = xm.xla_device()
        return DeviceInfo(
            backend="tpu",
            device=dev,
            name=str(dev),
            is_tpu=True,
            xla_model=xm,
        )
    except Exception:
        return None


def detect_device(prefer: str = "auto") -> DeviceInfo:
    """Detect CPU/GPU/TPU runtime.

    Resolution order for ``prefer='auto'``:
    1. TPU via torch_xla if available and usable.
    2. CUDA GPU.
    3. Apple MPS GPU.
    4. CPU.

    Explicit ``prefer`` can be one of: auto, tpu, cuda, gpu, mps, cpu.
    """
    prefer = (prefer or "auto").lower()
    if prefer in {"auto", "tpu", "xla"}:
        tpu = _try_tpu()
        if tpu is not None:
            return tpu
        if prefer in {"tpu", "xla"}:
            raise RuntimeError("TPU requested, but torch_xla/XLA device was not available.")

    if prefer in {"auto", "cuda", "gpu"} and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        return DeviceInfo(
            backend="cuda",
            device=torch.device("cuda"),
            name=torch.cuda.get_device_name(idx),
            is_cuda=True,
        )
    if prefer in {"cuda", "gpu"}:
        raise RuntimeError("CUDA/GPU requested, but torch.cuda.is_available() is False.")

    if prefer in {"auto", "mps"} and getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return DeviceInfo(
                backend="mps",
                device=torch.device("mps"),
                name="Apple Metal Performance Shaders",
                is_mps=True,
            )
    if prefer == "mps":
        raise RuntimeError("MPS requested, but MPS is not available.")

    return DeviceInfo(
        backend="cpu",
        device=torch.device("cpu"),
        name=f"CPU / torch {torch.__version__}",
        is_cpu=True,
    )


class HtSDeviceManager:
    """Small device abstraction for CPU/CUDA/MPS/TPU.

    It keeps the training code mostly identical across runtimes. TPU support is
    intentionally optional; when torch_xla is installed, optimizer steps call
    ``xm.optimizer_step`` and ``xm.mark_step``.
    """

    def __init__(self, prefer: str = "auto") -> None:
        self.info = detect_device(prefer)
        self.device = self.info.device

    @property
    def backend(self) -> str:
        return self.info.backend

    def to_device(self, obj):
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.to_device(x) for x in obj)
        if isinstance(obj, dict):
            return {k: self.to_device(v) for k, v in obj.items()}
        if hasattr(obj, "to"):
            return obj.to(self.device)
        return obj

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.info.is_tpu and self.info.xla_model is not None:
            self.info.xla_model.optimizer_step(optimizer)
            self.info.xla_model.mark_step()
        else:
            optimizer.step()

    def synchronize(self) -> None:
        if self.info.is_cuda:
            torch.cuda.synchronize()
        elif self.info.is_tpu and self.info.xla_model is not None:
            self.info.xla_model.mark_step()

    def autocast(self, enabled: bool = False):
        # Safe no-op default. CUDA AMP can be enabled by user externally.
        from contextlib import nullcontext
        if enabled and self.info.is_cuda:
            return torch.cuda.amp.autocast()
        return nullcontext()
