import pytest
from hts.device import detect_device, HtSDeviceManager, DeviceInfo


class TestDetectDevice:
    def test_cpu_detection(self):
        info = detect_device("cpu")
        assert info.backend == "cpu"
        assert info.is_cpu is True
        assert info.is_cuda is False
        assert info.is_mps is False
        assert info.is_tpu is False

    def test_auto_detection(self):
        info = detect_device("auto")
        assert info.backend in {"cpu", "cuda", "mps", "tpu"}

    def test_invalid_device_raises(self):
        with pytest.raises(RuntimeError, match="CUDA/GPU requested"):
            detect_device("cuda")

    def test_tpu_not_available(self):
        with pytest.raises(RuntimeError, match="TPU requested"):
            detect_device("tpu")

    def test_device_info_str(self):
        info = detect_device("cpu")
        s = str(info)
        assert "backend=cpu" in s


class TestHtSDeviceManager:
    def test_cpu_manager(self):
        mgr = HtSDeviceManager("cpu")
        assert mgr.backend == "cpu"
        assert str(mgr.device) == "cpu"

    def test_to_device_tensor(self):
        mgr = HtSDeviceManager("cpu")
        import torch
        x = torch.randn(4, 8)
        x_dev = mgr.to_device(x)
        assert str(x_dev.device) == "cpu"

    def test_to_device_dict(self):
        mgr = HtSDeviceManager("cpu")
        import torch
        d = {"a": torch.randn(4), "b": torch.randn(8)}
        d_dev = mgr.to_device(d)
        assert str(d_dev["a"].device) == "cpu"
        assert str(d_dev["b"].device) == "cpu"

    def test_to_device_list(self):
        mgr = HtSDeviceManager("cpu")
        import torch
        lst = [torch.randn(4), torch.randn(8)]
        lst_dev = mgr.to_device(lst)
        assert str(lst_dev[0].device) == "cpu"

    def test_optimizer_step(self):
        mgr = HtSDeviceManager("cpu")
        import torch
        model = torch.nn.Linear(8, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        mgr.optimizer_step(opt)

    def test_synchronize(self):
        mgr = HtSDeviceManager("cpu")
        mgr.synchronize()

    def test_autocast_disabled(self):
        mgr = HtSDeviceManager("cpu")
        ctx = mgr.autocast(enabled=False)
        assert ctx is not None
