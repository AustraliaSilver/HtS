"""Minimal HtS-B12 inference/training-style forward example."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from hts_b12 import HtSB12Classifier, HtSB12Config, HtSB12Objective, count_parameters, detect_device
from hts_b12.data import make_string_count_batch

info = detect_device("auto")
config = HtSB12Config(vocab_size=128, num_tasks=8, num_classes=64, max_length=32, d_model=64, dim_ff=128, num_layers=1)
model = HtSB12Classifier(config).to(info.device)
objective = HtSB12Objective(warmup_steps=10)

batch = make_string_count_batch(batch_size=8, max_length=32, device=info.device, num_classes=config.num_classes, seed=123)
logits = model(batch.input_ids, batch.task_ids, batch.attention_mask)
loss = objective(model, logits, batch.labels, step=1)

print(f"device={info.backend}:{info.device}")
print(f"params={count_parameters(model):,}")
print(f"logits={tuple(logits.shape)} loss={float(loss.loss.detach()):.4f}")
print(model.hts_diagnostics())
