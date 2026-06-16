# HtS-B12

HtS-B12 is a PyTorch research library for **Hard-to-Soft generated computation**. It exposes a Transformer-like sequence-classification API while adding task-conditioned, low-rank generated updates inside the model's real feed-forward path.

The package is intended for controlled experiments, custom task groups, and benchmark comparisons against a parameter-matched Transformer baseline.

## Highlights

- `HtSB12Classifier`: task-conditioned sequence classifier with HtS-B12 FFN updates.
- `TransformerClassifier`: baseline model with a matching input API.
- `ModelGroupConfig`: declarative task/label/vocabulary configuration.
- `ModelGroupRegistry`: register custom groups with batch factories.
- YAML/JSON group configs for reproducible experiments.
- CLI entry point for quick training runs.
- CPU, CUDA, Apple MPS, and optional TPU detection.
- Diagnostics for HtS gates, delta ratios, correction ratios, and regularizers.
- Benchmark scripts for string/count and publication-style validation protocols.

## Repository Layout

```text
src/hts_b12/      Python package
benchmarks/       Reproducible experiment scripts
configs/          Example model-group YAML files
examples/         Minimal API examples
docs/             Extended benchmark and usage notes
tests/            Smoke tests
```

## Installation

Use editable install from the repository root:

```bash
pip install -e .
```

For development:

```bash
pip install -e .[dev]
pytest -q
```

Optional extras:

```bash
pip install -e .[plot]   # plotting/report helpers
pip install -e .[tpu]    # torch-xla environments
```

## Quick Start

Create a built-in string length/count model:

```python
from hts_b12 import HtSB12Classifier, build_hts_config_from_group, string_length_count_group

group = string_length_count_group(max_length=64, num_classes=128)
config = build_hts_config_from_group(group)
model = HtSB12Classifier(config)

print(model)
```

Run a minimal forward/loss example:

```bash
python examples/quick_start.py
```

Train HtS-B12 on the built-in synthetic string/count task:

```bash
hts-b12 --model hts-b12 --device auto --steps 1000 --batch-size 64
```

Train the Transformer baseline with the same CLI surface:

```bash
hts-b12 --model transformer --device auto --steps 1000 --batch-size 64
```

The CLI writes `train_log.csv` and the best checkpoint, when enabled, under `runs/hts_b12_demo` by default.

## Model Groups

A model group defines the task family, vocabulary, sequence length, labels, and recommended model/training settings. Groups let the same HtS-B12 code run beyond the built-in string/count task.

```python
from hts_b12 import LabelSpec, ModelGroupConfig, TaskSpec

group = ModelGroupConfig(
    name="toy_parity",
    vocab_size=32,
    max_length=32,
    tasks=[
        TaskSpec("sum_parity", 0, "Predict whether the token sum is even or odd."),
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

Build a model from the group:

```python
from hts_b12 import HtSB12Classifier, build_hts_config_from_group

config = build_hts_config_from_group(group)
model = HtSB12Classifier(config)
```

Override selected model settings directly:

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

## Custom Batches

Training functions consume batches with this shape:

```python
from hts_b12 import HtSBatch

HtSBatch(
    input_ids=...,       # LongTensor [batch, seq_len]
    task_ids=...,        # LongTensor [batch]
    labels=...,          # LongTensor [batch]
    attention_mask=...,  # optional LongTensor [batch, seq_len], 1 = valid token
)
```

Example custom batch factory:

```python
import torch
from hts_b12 import HtSBatch


def make_parity_batch(batch_size: int, device: torch.device, seed: int) -> HtSBatch:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    max_length = 32
    lengths = torch.randint(4, max_length + 1, (batch_size,), generator=generator)

    input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.long)
    task_ids = torch.zeros(batch_size, dtype=torch.long)

    for index, length in enumerate(lengths.tolist()):
        tokens = torch.randint(1, 20, (length,), generator=generator)
        input_ids[index, :length] = tokens
        attention_mask[index, :length] = 1
        labels[index] = int(tokens.sum().item() % 2)

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
from hts_b12 import ModelGroupRegistry, TrainConfig, build_hts_config_from_group
from hts_b12 import HtSB12Classifier
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

See `examples/custom_group.py` for a complete example.

## YAML Group Config

Groups can also be stored as YAML:

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

Load it in Python:

```python
from hts_b12 import ModelGroupConfig, build_hts_config_from_group

group = ModelGroupConfig.load("configs/string_length_count.yaml")
config = build_hts_config_from_group(group)
```

Use it from the CLI:

```bash
hts-b12 --group-config configs/string_length_count.yaml --steps 1000
```

Important: `--group-config` changes the model/group shape. The CLI still uses the built-in string/count synthetic batch factory. For real custom datasets, use the Python registry API.

## Device Selection

```python
from hts_b12 import detect_device

info = detect_device("auto")
print(info.backend)  # cpu, cuda, mps, or tpu
print(info.device)
print(info.name)
```

Supported preferences:

- `auto`: TPU, CUDA, MPS, then CPU.
- `cpu`: force CPU.
- `cuda` or `gpu`: force NVIDIA CUDA.
- `mps`: force Apple Metal Performance Shaders.
- `tpu` or `xla`: force `torch_xla`.

## Architecture

HtS-B12 modifies the Transformer FFN path with task-conditioned generated updates:

```text
h = GELU(W1(x) + main_delta(x, task) + correction_delta(x, task))
y = W2(h) + main_delta(h, task)
```

The generated updates are controlled by:

- task embeddings;
- input context;
- ratio-targeted delta scaling;
- correction branch;
- task-specific router offsets;
- margin-oriented objective.

The core idea is:

```text
hard controller -> generated soft computation -> end-to-end task loss
```

## Save and Load

```python
model.save_pretrained("runs/my_hts_model")

from hts_b12 import HtSB12Classifier

model = HtSB12Classifier.from_pretrained("runs/my_hts_model")
```

## Diagnostics

```python
diag = model.hts_diagnostics()
for key, value in diag.items():
    print(key, value)
```

Common diagnostic groups:

- main gate and correction gate;
- delta/base ratio and correction ratio;
- target ratio values;
- task offset magnitude;
- generated rank coefficient statistics;
- regularizer proxies.

## Benchmarks

Run the simple string/count benchmark:

```bash
python benchmarks/string_length_count.py
```

Run the patched validation benchmark:

```bash
python benchmarks/kaggle_validation_a.py --quick
python benchmarks/kaggle_validation_a.py --steps 2000 --seeds 42,123,777
```

Run publication-style Benchmark-C:

```bash
python benchmarks/kaggle_publication_c.py --quick
python benchmarks/kaggle_publication_c.py --protocols c1,c2,c3 --steps 5000 --seeds 42,123,777
```

Run publication-style Benchmark-D with OOD fixes:

```bash
python benchmarks/kaggle_publication_d_oodfix.py --quick
```

For Kaggle dual-GPU runs:

```bash
python /kaggle/working/HtS/benchmarks/kaggle_publication_c_dualgpu.py --quick
python /kaggle/working/HtS/benchmarks/kaggle_publication_d_dualgpu.py --quick
```

See:

- `PATCH_NOTES.md`
- `docs/PUBLICATION_BENCHMARK.md`
- `docs/PUBLICATION_BENCHMARK_C.md`
- `docs/PUBLICATION_BENCHMARK_D_OODFIX.md`
- `docs/DUAL_GPU_KAGGLE.md`

## Recommended Training Defaults

For stable B12 training, start with:

```yaml
optimizer: AdamW
lr: 0.001
scheduler: cosine_decay
warmup_steps: 250
grad_clip: 1.0
weight_decay: 0.01
save_best: true
```

Avoid high learning rates such as `3e-3` until stability is verified across multiple seeds.

## Fair Benchmarking Checklist

When comparing HtS-B12 to the Transformer baseline:

1. Use the same train/dev/test split.
2. Compare same-parameter and same-FLOP baselines.
3. Run multiple seeds.
4. Keep held-out length/task distributions untouched.
5. Report loss, accuracy, margin, calibration, and parameter count.
6. Construct models after seeding.
7. Evaluate the best validation checkpoint on the final test split.
8. Freeze configs before final held-out evaluation.

## Known Limitations

- `train_classifier` reports accuracy on the current generated batch; benchmark scripts should use explicit validation/test splits for publication claims.
- The CLI is intentionally simple and does not load arbitrary custom datasets.
- `ModelGroupRegistry` is in-process only; production workflows should own their dataset loading and experiment tracking.
- HtS-B12 is research code. Treat benchmark wins and failures as experimental evidence, not guaranteed model behavior.

## Code Quality Notes

Current strengths:

- Clear `src/` package layout with typed dataclass configs.
- Small public API exported from `hts_b12.__init__`.
- Deterministic synthetic batch generation through explicit seeds.
- Useful diagnostics for HtS-specific behavior.
- Smoke tests cover forward shape, group config construction, and registry factories.

Recommended next fixes:

- Add a true validation batch function to `train_classifier` instead of selecting best checkpoints from the training batch.
- Mask mean pooling in `TransformerClassifier` when `pool="mean"` and `attention_mask` is provided.
- Add config validation for invalid task IDs, duplicate task IDs, and incompatible dimensions such as `d_model % n_heads != 0`.
- Replace the deterministic modulo sampler in `make_multi_group_factory` with generator-based random sampling.
- Expand tests around attention masks, save/load round trips, CLI execution, and YAML config loading.

## License

MIT License. See `LICENSE`.
