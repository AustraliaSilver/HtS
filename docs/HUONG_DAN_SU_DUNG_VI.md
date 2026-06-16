# Hướng dẫn sử dụng HtS-B12

Thư viện này không chỉ train một benchmark cố định. Bạn có thể tự tạo **nhóm mô hình / nhóm task** bằng `ModelGroupConfig`.

## 1. Cài đặt

```bash
cd hts_b12_library
pip install -e .
```

## 2. Chạy nhanh

```bash
hts-b12 --model hts-b12 --device auto --steps 1000 --batch-size 64
```

## 3. Tạo nhóm task riêng

```python
from hts_b12 import ModelGroupConfig, TaskSpec, LabelSpec

group = ModelGroupConfig(
    name="toy_parity",
    vocab_size=32,
    max_length=32,
    tasks=[TaskSpec("sum_parity", 0, "Dự đoán tổng token là chẵn/lẻ")],
    labels=LabelSpec(num_classes=2, names=["even", "odd"]),
    recommended_model={
        "d_model": 64,
        "n_heads": 4,
        "num_layers": 1,
        "dim_ff": 128,
        "rank_main": 4,
        "rank_corr": 2,
    },
)
```

## 4. Tạo batch factory

```python
from hts_b12 import HtSBatch

def batch_factory(batch_size, device, seed):
    # Trả về input_ids, task_ids, labels, attention_mask
    return HtSBatch(input_ids, task_ids, labels, attention_mask)
```

## 5. Đăng ký và train

```python
from hts_b12 import ModelGroupRegistry, HtSB12Classifier, build_hts_config_from_group, TrainConfig
from hts_b12.training import train_group_classifier

registry = ModelGroupRegistry()
registry.register(group, batch_factory)

model = HtSB12Classifier(build_hts_config_from_group(group))
log = train_group_classifier(model, "toy_parity", TrainConfig(steps=1000), registry=registry)
```

## 6. Dùng file YAML

```bash
hts-b12 --group-config configs/string_length_count.yaml --steps 1000
```

Với dataset thật, nên dùng Python API để tự viết batch factory.

## 7. Thiết bị

```python
from hts_b12 import detect_device
info = detect_device("auto")
print(info.backend)  # cpu, cuda, mps, tpu
```

## 8. Khuyến nghị train

```yaml
lr: 0.001
warmup_steps: 250
scheduler: cosine_decay
grad_clip: 1.0
weight_decay: 0.01
save_best: true
```

Không nên dùng learning rate quá cao nếu chưa kiểm tra stability.
