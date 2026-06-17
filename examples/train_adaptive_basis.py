"""Train HtS-B12 with AdaptiveBasisLowRank on synthetic data (quick CPU test)."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from hts_b12 import HtSB12Classifier, HtSB12Config, HtSB12Objective, TrainConfig, accuracy, count_parameters
from hts_b12.training import train_classifier
from hts_b12.groups import HtSBatch

B = 32
VOCAB = 32
N_TASKS = 2
N_CLASSES = 16
MAX_LEN = 16
STEPS = 100
EVAL = 20
DEVICE = "cpu"

def make_batch(batch_size, device, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    length = torch.randint(4, MAX_LEN + 1, (batch_size,), generator=gen)
    ids = torch.randint(1, VOCAB, (batch_size, MAX_LEN), generator=gen)
    pos = torch.arange(MAX_LEN).unsqueeze(0)
    mask = pos < length.unsqueeze(1)
    ids = ids.masked_fill(~mask, 0)
    task = torch.randint(0, N_TASKS, (batch_size,), generator=gen)
    label = (ids == 1).sum(dim=1).clamp_max(N_CLASSES - 1)
    return HtSBatch(
        input_ids=ids.to(device),
        task_ids=task.to(device),
        labels=label.to(device),
        attention_mask=mask.long().to(device),
    )

cfg = HtSB12Config(
    vocab_size=VOCAB, num_tasks=N_TASKS, num_classes=N_CLASSES, max_length=MAX_LEN,
    d_model=32, n_heads=4, num_layers=2, dim_ff=64,
    task_dim=8, rank_main=[8, 16], rank_corr=[4, 8], rank_task_attn=[4, 8],
    dropout=0.0, use_cls_token=True, pool="cls",
    alpha_max=1.05, target_min=0.2, target_max=0.9,
    corr_alpha_max=0.55, corr_gain=6.0, task_offset_scale=0.3,
    ratio_ceiling=2.0, corr_ceiling=1.0, router_per_task=True,
)

model = HtSB12Classifier(cfg)
print(f"Params: {count_parameters(model):,}")

train_cfg = TrainConfig(
    steps=STEPS, batch_size=B, lr=1e-3, weight_decay=1e-2,
    warmup_steps=10, grad_clip=1.0, eval_every=EVAL,
    device=DEVICE, seed=42, save_best=False,
)

log = train_classifier(model, make_batch, train_cfg)
print(f"\nBest validation accuracy: {log.best_acc*100:.2f}%")
for row in log.rows:
    diag = {k: v for k, v in row.items() if isinstance(v, float) and ("rank" in k or "coeff" in k or "delta" in k or "entropy" in k or "gate" in k or "alpha" in k)}
    print(f"step={row['step']} acc={row['accuracy']*100:.1f}% loss={row['eval_loss']:.4f}", end="")
    if diag:
        items = " ".join(f"{k}={v:.3f}" for k, v in sorted(diag.items()))
        print(f" [{items}]")
    else:
        print()
