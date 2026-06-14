# HtS Foundation — Hard-to-Soft Generated Computation Library

`hts-foundation` is a clean Python/PyTorch project that packages the final CPU-scale HtS optimization line into a reusable architecture library.

The current reference architecture is **HtS-B12**:

- true FFN soft-weight generation;
- ratio-controlled generated updates;
- task-specific router offsets;
- small input-side correction branch;
- adaptive rank masks;
- margin-oriented objective;
- CPU/GPU/TPU device detection.

This project is designed as a research library, not a production LLM framework.

---

## 1. Core idea

A standard Transformer uses mostly static learned linear maps:

```text
Y = X W0
```

HtS adds generated, task/input-conditioned soft computation:

```text
Y = X W0 + generated_soft_update(X, task)
```

The B12 FFN update is conceptually:

```text
base1 = X W1
main1 = gate(x,t) * alpha(x,t) * ratio_normalize(raw_delta1, base1, target_ratio1)
corr1 = corr_gate(x,t) * corr_alpha(x,t) * free_correction1
h = GELU(base1 + main1 + corr1)

base2 = h W2
main2 = gate(x,t) * alpha(x,t) * ratio_normalize(raw_delta2, base2, target_ratio2)
Y = base2 + main2
```

Each raw delta is generated as a diagonal low-rank soft-weight map:

```text
delta = (X A^T) diag(m(task) * (s_task(task) + s_tune(input, task))) B^T
```

The generated weights are **not direct trainable parameters**. They are produced during forward pass by hard generator/router weights and optimized only through task loss.

---

## 2. Project structure

```text
hts_foundation_project/
├── hts/
│   ├── __init__.py
│   ├── config.py              # HtSConfig, TransformerConfig, TrainConfig
│   ├── device.py              # CPU/GPU/TPU detection and device manager
│   ├── layers.py              # GeneratedDiagonalLinear, HtSB12FFN
│   ├── models.py              # HtSTransformerClassifier, StaticTransformerClassifier
│   ├── losses.py              # CE + margin + HtS regularization
│   ├── diagnostics.py         # accuracy and router diagnostics helpers
│   ├── training.py            # train_synthetic and evaluate
│   ├── cli.py                 # command-line interface
│   └── data/
│       └── synthetic_tasks.py # synthetic task-conditioned benchmark
├── examples/
│   ├── train_synthetic.py
│   └── compare_hts_transformer.py
├── scripts/
│   ├── run_cpu.sh
│   ├── run_gpu.sh
│   └── run_tpu.sh
├── tests/
│   └── test_smoke.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 3. Installation

### CPU/GPU environment

```bash
cd hts_foundation_project
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### TPU environment

TPU support uses optional PyTorch/XLA. Install it only inside a TPU runtime:

```bash
pip install -e .
pip install torch-xla
```

The package does not require `torch-xla` on CPU/GPU.

---

## 4. Device detection

```python
from hts import detect_device, HtSDeviceManager

info = detect_device("auto")
print(info.backend)  # tpu, cuda, mps, or cpu
print(info.name)

manager = HtSDeviceManager("auto")
```

Detection order for `auto`:

1. TPU via `torch_xla`, if available;
2. CUDA GPU;
3. Apple MPS GPU;
4. CPU.

Explicit choices:

```python
detect_device("cpu")
detect_device("cuda")
detect_device("mps")
detect_device("tpu")
```

On TPU, optimizer steps are routed through `xm.optimizer_step(optimizer)` automatically by `HtSDeviceManager.optimizer_step`.

---

## 5. Quick training

### Train HtS on auto device

```bash
python -m hts.cli --model hts --device auto --steps 300 --batch-size 64 --out-dir runs/hts_auto
```

### Train Transformer baseline

```bash
python -m hts.cli --model transformer --device auto --steps 300 --batch-size 64 --out-dir runs/transformer_auto
```

### CPU script

```bash
bash scripts/run_cpu.sh
```

### GPU script

```bash
bash scripts/run_gpu.sh
```

### TPU script

```bash
bash scripts/run_tpu.sh
```

---

## 6. Python API

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
        steps=500,
        batch_size=128,
        device="auto",
    ),
    out_dir="runs/my_hts_run",
)

print(result["meta"])
print(result["metrics_path"])
```

---

## 7. Model usage

```python
import torch
from hts import HtSConfig
from hts.models import HtSTransformerClassifier
from hts.data.synthetic_tasks import SyntheticTaskBatcher

model = HtSTransformerClassifier(HtSConfig())
batcher = SyntheticTaskBatcher()
batch = batcher.batch(16, family="arith8")

logits = model(batch["input_ids"], batch["task_ids"])
print(logits.shape)
print(model.diagnostics())
```

---

## 8. Diagnostics

HtS exposes diagnostics after a forward pass:

```python
diag = model.diagnostics()
for key, value in diag.items():
    print(key, value)
```

Typical fields:

```text
block0.hts_l0_gate_main
block0.hts_l0_alpha_main
block0.hts_l0_target1
block0.hts_l0_target2
block0.hts_l0_delta_base_ratio
block0.hts_l0_corr_ratio
block0.hts_l0_budget
block0.hts_l0_main1_rank_eff
block0.hts_l0_corr1_rank_eff
```

Important interpretation:

- `delta_base_ratio`: how strongly generated computation affects the FFN compared with the static base path.
- `budget`: approximate generated-computation usage.
- `rank_eff`: effective soft rank used by generated diagonal map.
- `gate_main` and `gate_corr`: router activation strength.

---

## 9. Losses

The default training objective is:

```text
L = CE + margin_weight * margin_loss
    + budget_weight * budget
    + binary_weight * gate_binary_penalty
    + ratio_weight * ratio_penalty
    + task_offset_weight * task_offset_l2
```

The margin term encourages the correct logit to exceed the strongest wrong logit. This was introduced after B11 because lower loss did not always translate to higher accuracy.

---

## 10. Current research status

This package represents the **B12 CPU-scale endpoint** of the HtS optimization process.

Best CPU-scale findings so far:

- Generated computation must be placed inside the true FFN path, not merely in the output head.
- Soft updates must have measurable delta/base ratio; near-zero generated branches are misleading.
- Ratio-controlled updates improve loss/calibration.
- Task-specific router offsets plus margin loss gave the best CPU-scale macro accuracy.
- Further small CPU variants are not recommended; the next meaningful step is GPU/TPU-scale validation.

Recommended next experiment:

```bash
python -m hts.cli --model hts --device cuda --steps 10000 --batch-size 512 --eval-every 500 --out-dir runs/hts_b12_gpu
python -m hts.cli --model transformer --device cuda --steps 10000 --batch-size 512 --eval-every 500 --out-dir runs/transformer_gpu
```

---

## 11. Notes and limitations

- TPU support is optional and depends on a valid `torch_xla` runtime.
- The included benchmark is synthetic and compact.
- This is not yet evidence that HtS is a Transformer-level architecture.
- The project is structured to make GPU/TPU scaling and new benchmarks easier.

---

## 12. Minimal smoke test

```bash
python -m pytest tests/test_smoke.py
```

Or without pytest:

```bash
python examples/train_synthetic.py
```
