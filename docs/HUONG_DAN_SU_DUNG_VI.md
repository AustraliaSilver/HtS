# Hướng dẫn sử dụng thư viện HtS Foundation

## 1. Mục tiêu thư viện

Thư viện này đóng gói kiến trúc **HtS-B12** thành một dự án Python/PyTorch hoàn chỉnh. HtS ở đây được hiểu là **Hard-to-Soft Generated Computation**: mô hình có một phần hard weights học cách tạo ra soft-weight updates theo task và input.

Thư viện hỗ trợ:

- chạy trên CPU;
- chạy trên GPU CUDA;
- chạy trên Apple MPS;
- chạy trên TPU nếu môi trường có `torch_xla`;
- huấn luyện benchmark synthetic task-conditioned;
- so sánh với Transformer gốc;
- xuất metrics, diagnostics và checkpoint.

---

## 2. Cài đặt

```bash
cd hts_foundation_project
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Nếu chỉ muốn chạy nhanh không cài package:

```bash
cd hts_foundation_project
PYTHONPATH=. python -m hts.cli --model hts --device cpu --steps 10
```

---

## 3. Kiểm tra thiết bị

```python
from hts import detect_device

info = detect_device("auto")
print(info.backend)
print(info.name)
```

Kết quả có thể là:

```text
cpu
cuda
mps
tpu
```

Thứ tự tự động:

1. TPU nếu có `torch_xla`;
2. CUDA GPU nếu có NVIDIA GPU;
3. Apple MPS nếu chạy trên Mac hỗ trợ Metal;
4. CPU nếu không có accelerator.

---

## 4. Train HtS nhanh trên CPU

```bash
python -m hts.cli \
  --model hts \
  --device cpu \
  --steps 300 \
  --batch-size 64 \
  --out-dir runs/hts_cpu
```

Nếu CPU bị chậm do quá nhiều thread, dùng:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m hts.cli --model hts --device cpu --steps 300
```

---

## 5. Train trên GPU

```bash
python -m hts.cli \
  --model hts \
  --device cuda \
  --steps 10000 \
  --batch-size 512 \
  --eval-every 500 \
  --out-dir runs/hts_gpu
```

Train Transformer baseline:

```bash
python -m hts.cli \
  --model transformer \
  --device cuda \
  --steps 10000 \
  --batch-size 512 \
  --eval-every 500 \
  --out-dir runs/transformer_gpu
```

---

## 6. Train trên TPU

Cần môi trường TPU có PyTorch/XLA:

```bash
pip install torch-xla
```

Chạy:

```bash
python -m hts.cli \
  --model hts \
  --device tpu \
  --steps 10000 \
  --batch-size 512 \
  --eval-every 500 \
  --out-dir runs/hts_tpu
```

Trong code, `HtSDeviceManager` sẽ tự dùng `xm.optimizer_step(optimizer)` khi phát hiện TPU.

---

## 7. Dùng bằng Python API

```python
from hts import HtSConfig, TrainConfig
from hts.training import train_synthetic

result = train_synthetic(
    model_kind="hts",
    hts_config=HtSConfig(
        d_model=40,
        dim_ff=64,
        rank_main=5,
        rank_corr=2,
    ),
    train_config=TrainConfig(
        steps=1000,
        batch_size=128,
        device="auto",
    ),
    out_dir="runs/my_hts",
)

print(result["metrics_path"])
```

---

## 8. Tùy chỉnh kiến trúc HtS

```python
from hts import HtSConfig

cfg = HtSConfig(
    d_model=40,
    dim_ff=64,
    n_layers=1,
    rank_main=5,
    rank_corr=2,
    alpha_max=1.18,
    target_min=0.34,
    target_max=0.90,
    task_offset_scale=0.30,
)
```

Ý nghĩa các tham số chính:

| Tham số | Ý nghĩa |
|---|---|
| `d_model` | kích thước hidden representation |
| `dim_ff` | kích thước FFN |
| `rank_main` | rank của soft-weight update chính |
| `rank_corr` | rank của correction branch |
| `alpha_max` | cường độ tối đa của generated update |
| `target_min`, `target_max` | miền kiểm soát delta/base ratio |
| `task_offset_scale` | mức task-specific router offset |
| `correction_mode` | đặt correction ở input/output/both/none |

---

## 9. Đọc diagnostics

Sau một forward pass:

```python
print(model.diagnostics())
```

Các chỉ số quan trọng:

| Chỉ số | Ý nghĩa |
|---|---|
| `gate_main` | mức bật nhánh generated chính |
| `alpha_main` | cường độ nhánh generated chính |
| `target1`, `target2` | target delta/base ratio |
| `delta_base_ratio` | soft update mạnh bao nhiêu so với base FFN |
| `corr_ratio` | correction branch mạnh bao nhiêu |
| `budget` | ngân sách generated computation |
| `rank_eff` | rank hiệu dụng của soft map |

Nếu `delta_base_ratio` quá gần 0, generated computation gần như không có tác dụng. Đây là lỗi đã từng xảy ra ở B5/B6.

---

## 10. Kết luận nghiên cứu được đóng gói

Phiên bản thư viện hiện tại giữ kết luận B12:

- không dùng dynamic output head làm trung tâm;
- không dùng deliberation loop yếu như B5;
- đặt generated soft weights trực tiếp trong true FFN path;
- dùng ratio-control để soft update có tác động thật nhưng không phá base path;
- thêm task-specific router offsets;
- thêm margin loss để tăng accuracy.

Bước tiếp theo đúng nghĩa là benchmark GPU/TPU dài hơn, không phải thêm biến thể CPU nhỏ.
