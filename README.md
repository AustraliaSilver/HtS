# HtS-B12 Library

**HtS-B12** is a PyTorch library for **Hard-to-Soft generated computation**.  It is designed to feel direct like a Transformer library, while exposing the core HtS idea: a hard controller generates soft, task-conditioned weight updates inside the model's true FFN computation path.

This version is **not locked to one benchmark**.  Users can define their own **model groups**: task families, label spaces, vocabulary sizes, training settings, and batch adapters.

---

## Key Features

- `HtSB12Classifier`: task-conditioned sequence classifier using HtS-B12 FFN updates.
- `TransformerClassifier`: baseline with matching input API.
- `ModelGroupConfig`: define your own task/model groups.
- `ModelGroupRegistry`: register groups and batch factories.
- YAML/JSON group configs.
- Direct Python API for custom datasets.
- CLI for quick training and inspection.
- CPU/GPU/MPS/TPU auto detection.
- Margin-oriented objective and HtS diagnostics.
- MIT License.

---

## Installation

```bash
cd hts_b12_library
pip install -e .
```

For development:

```bash
pip install -e .[dev]
pytest -q
```

---

## Quick Start

```python
from hts_b12 import HtSB12Classifier, build_hts_config_from_group, string_length_count_group

# Built-in group preset.
group = string_length_count_group(max_length=64, num_classes=128)
config = build_hts_config_from_group(group)
model = HtSB12Classifier(config)

print(model)
```

Train the built-in string/count task:

```bash
hts-b12 --model hts-b12 --device auto --steps 1000 --batch-size 64
```

Train Transformer baseline:

```bash
hts-b12 --model transformer --device auto --steps 1000 --batch-size 64
```

---

## The Important Part: Model Groups

A **model group** is a user-defined family of tasks.  Instead of hard-coding only `length` or `count`, you can define any group:

- string length/counting;
- chess move objectives;
- symbolic arithmetic;
- classification groups;
- custom routing tasks;
- instruction-conditioned tasks.

### Define a Group in Python

```python
from hts_b12 import ModelGroupConfig, TaskSpec, LabelSpec

group = ModelGroupConfig(
    name="toy_parity",
    vocab_size=32,
    max_length=32,
    tasks=[
        TaskSpec("sum_parity", 0, "Predict whether token sum is even or odd."),
    ],
    labels=LabelSpec(num_classes=2, names=["even", "odd"]),
    recommended_model={
        "d_model": 64,
        "n_heads": 4,
        "num_layers": 1,
        "dim_ff": 128,
        "rank_main": 4,
        "rank_corr": 2,
        "task_dim": 16,
    },
    recommended_training={
        "lr": 1e-3,
        "warmup_steps": 100,
        "grad_clip": 1.0,
    },
)
```

### Build a Model from the Group

```python
from hts_b12 import HtSB12Classifier, build_hts_config_from_group

config = build_hts_config_from_group(group)
model = HtSB12Classifier(config)
```

Override model size directly:

```python
config = build_hts_config_from_group(
    group,
    d_model=256,
    num_layers=4,
    dim_ff=512,
    rank_main=12,
    rank_corr=6,
)
```

---

## Custom Batch Factory

HtS expects batches in a standard form:

```python
from hts_b12 import HtSBatch

HtSBatch(
    input_ids=...,       # LongTensor [B, T]
    task_ids=...,        # LongTensor [B]
    labels=...,          # LongTensor [B]
    attention_mask=...,  # optional LongTensor [B, T]
)
```

Example custom data factory:

```python
import torch
from hts_b12 import HtSBatch


def make_parity_batch(batch_size: int, device: torch.device, seed: int) -> HtSBatch:
    g = torch.Generator(device="cpu").manual_seed(seed)
    max_len = 32
    lengths = torch.randint(4, max_len + 1, (batch_size,), generator=g)

    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.long)
    task_ids = torch.zeros(batch_size, dtype=torch.long)

    for i, L in enumerate(lengths.tolist()):
        toks = torch.randint(1, 20, (L,), generator=g)
        input_ids[i, :L] = toks
        attention_mask[i, :L] = 1
        labels[i] = int(toks.sum().item() % 2)

    return HtSBatch(
        input_ids=input_ids.to(device),
        task_ids=task_ids.to(device),
        labels=labels.to(device),
        attention_mask=attention_mask.to(device),
        group="toy_parity",
    )
```

Register and train:

```python
from hts_b12 import ModelGroupRegistry, TrainConfig
from hts_b12.training import train_group_classifier

registry = ModelGroupRegistry()
registry.register(group, make_parity_batch)

model = HtSB12Classifier(build_hts_config_from_group(group))
log = train_group_classifier(
    model,
    group_name="toy_parity",
    config=TrainConfig(steps=1000, batch_size=64, device="auto"),
    registry=registry,
)
```

See full example: `examples/custom_group.py`.

---

## YAML Group Config

You can define groups in YAML:

```yaml
name: string_length_count
vocab_size: 128
max_length: 64
tasks:
  - {name: length, id: 0, description: Predict sequence length}
  - {name: count_a, id: 1, description: Count token a}
labels:
  num_classes: 128
recommended_model:
  d_model: 128
  n_heads: 4
  num_layers: 2
  dim_ff: 256
  rank_main: 8
  rank_corr: 4
recommended_training:
  lr: 0.001
  warmup_steps: 250
```

Load it:

```python
from hts_b12 import ModelGroupConfig, build_hts_config_from_group

group = ModelGroupConfig.load("configs/string_length_count.yaml")
config = build_hts_config_from_group(group)
```

Use it from CLI:

```bash
hts-b12 --group-config configs/string_length_count.yaml --steps 1000
```

The CLI still uses the built-in string/count synthetic factory for quick use. For real custom datasets, use the Python registry API.

---

## Device Detection

```python
from hts_b12 import detect_device

info = detect_device("auto")
print(info.backend)  # cpu, cuda, mps, tpu
print(info.device)
print(info.name)
```

Supported backends:

- CPU;
- CUDA GPU;
- Apple MPS;
- TPU via `torch_xla` if installed.

---

## Architecture Summary

HtS-B12 modifies the Transformer FFN path:

```text
h = GELU(W1(x) + main_ratio_delta(x, task) + correction_delta(x, task))
y = W2(h) + main_ratio_delta(h, task)
```

The generated soft updates are controlled by:

- task embeddings;
- input context;
- ratio-controlled delta scaling;
- correction branch;
- task-specific router offsets;
- margin-oriented objective.

This follows the original HtS philosophy:

> hard controller → generated soft computation → trained end-to-end by task loss.

---

## Save and Load

```python
model.save_pretrained("runs/my_hts_model")

from hts_b12 import HtSB12Classifier
model = HtSB12Classifier.from_pretrained("runs/my_hts_model")
```

---

## Diagnostics

```python
diag = model.hts_diagnostics()
for k, v in diag.items():
    print(k, v)
```

Common diagnostics:

- main gate;
- correction gate;
- delta/base ratio;
- correction ratio;
- target ratio;
- task offset magnitude;
- generated rank coefficient stats.

---

## Recommended Training Defaults

For stable B12 training, prefer:

```yaml
optimizer: AdamW
lr: 0.001
scheduler: cosine_decay
warmup_steps: 250
grad_clip: 1.0
weight_decay: 0.01
save_best: true
```

Avoid overly high LR such as `3e-3` unless you have verified stability. In earlier string/count experiments, high LR caused a late loss spike.

---

## Fair Benchmarking Checklist

When comparing HtS to Transformer:

1. use same train/dev/test split;
2. compare same-params and same-FLOPs baselines;
3. run multiple seeds;
4. keep held-out length/task distributions untouched;
5. report loss, accuracy, margin, calibration and parameter count;
6. freeze config before final held-out evaluation.

---

## License

MIT License. See `LICENSE`.

## Publication Benchmark-C

This repository includes `benchmarks/kaggle_publication_c.py`, a diagnostic benchmark that follows Benchmark-B and tests whether held-out failure is caused by capacity/label availability, bucket interpolation, or true length extrapolation.

Run:

```bash
python benchmarks/kaggle_publication_c.py --quick
python benchmarks/kaggle_publication_c.py --protocols c1 --steps 5000 --seeds 42,123,777
python benchmarks/kaggle_publication_c.py --protocols c1,c2,c3 --steps 5000 --seeds 42,123,777
```

See `docs/PUBLICATION_BENCHMARK_C.md` for details.

## Kaggle T4×2 dual-GPU Benchmark-C

If your Kaggle runtime has two T4 GPUs, run HtS-B12 and the Transformer baseline concurrently:

```bash
python /kaggle/working/HtS/benchmarks/kaggle_publication_c_dualgpu.py \
  --protocols c1,c2,c3 \
  --steps 5000 \
  --seeds 42,123,777
```

The launcher assigns HtS-B12 to physical GPU `0` and Transformer-ParamMatched to physical GPU `1` via `CUDA_VISIBLE_DEVICES`, then merges results into `publication_c_results_dualgpu/`.

For a quick check:

```bash
python /kaggle/working/HtS/benchmarks/kaggle_publication_c_dualgpu.py --quick
```

See `docs/DUAL_GPU_KAGGLE.md`.
