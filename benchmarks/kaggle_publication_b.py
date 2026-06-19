"""Publication-grade Benchmark-B for HtS-B12 vs Transformer.

This script is designed for producing a clearer, more publishable result than
single-distribution accuracy alone.  It keeps the HtS-B12 configuration locked,
uses a parameter-matched Transformer baseline, avoids label clipping by setting
`num_classes > max_eval_length`, evaluates multiple held-out splits from the
same best-validation checkpoint, and optionally runs a no-soft-update ablation.

Recommended Kaggle commands
---------------------------
Main result, 3 seeds, HtS vs param-matched Transformer:
    !python /kaggle/working/HtS/benchmarks/kaggle_publication_b.py \
        --steps 5000 --seeds 42,123,777

Add no-soft-update ablation, slower:
    !python /kaggle/working/HtS/benchmarks/kaggle_publication_b.py \
        --steps 5000 --seeds 42,123,777 --include-ablation

Quick smoke:
    !python /kaggle/working/HtS/benchmarks/kaggle_publication_b.py --quick

The script writes:
    publication_b_results/summary.csv
    publication_b_results/all_runs.csv
    publication_b_results/result_card.md
    publication_b_results/config.json
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Always prefer local repo source over a stale installed package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from hts_b12 import HtSB12Classifier, HtSB12Config, TransformerClassifier, TrainConfig, accuracy, count_parameters
from hts_b12.losses import HtSB12Objective
from hts_b12.ood_heads import HtSB12DigitClassifier
from hts_b12.training import cosine_with_warmup

RESULTS_DIR = Path("publication_b_results")
DEFAULT_SEEDS = [42, 123, 777]

# Token ids are deliberately tiny and explicit.  PAD=0, real tokens 1..10.
PAD = 0
TOKENS = {
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "0": 5,
    "1": 6,
    "2": 7,
    "3": 8,
    "x": 9,
    "y": 10,
}
TASKS = {
    "length": 0,
    "count_a": 1,
    "count_b": 2,
    "count_digit": 3,
}


@dataclass
class Batch:
    input_ids: torch.Tensor
    task_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def make_publication_batch(
    batch_size: int,
    min_length: int,
    max_length: int,
    device: torch.device | str,
    seed: int | None,
    task_mix: Sequence[str],
    token_mode: str = "uniform",
    num_classes: int = 256,
) -> Batch:
    """Vectorized string length/count batch without silent label clipping.

    `num_classes` must be greater than the largest possible label.  The function
    raises instead of clipping because clipping would contaminate length/count
    generalization claims.
    """
    if max_length >= num_classes:
        raise ValueError(
            f"num_classes={num_classes} must exceed max_length={max_length}. "
            "Use --num-classes 256 for max_eval_length <= 255."
        )
    for task in task_mix:
        if task not in TASKS:
            raise KeyError(f"unknown task {task!r}; allowed={list(TASKS)}")

    gen = torch.Generator(device="cpu") if seed is not None else None
    if gen is not None:
        gen.manual_seed(int(seed))
    dev = torch.device(device)

    lengths = torch.randint(min_length, max_length + 1, (batch_size,), generator=gen, dtype=torch.long)

    if token_mode == "uniform":
        ids = torch.randint(1, len(TOKENS) + 1, (batch_size, max_length), generator=gen, dtype=torch.long)
    elif token_mode == "biased_count":
        # More a/b/digit tokens than in training.  Same tasks, shifted token distribution.
        values = torch.tensor([1, 1, 1, 2, 2, 5, 6, 7, 8, 9, 10, 3, 4], dtype=torch.long)
        idx = torch.randint(0, len(values), (batch_size, max_length), generator=gen, dtype=torch.long)
        ids = values[idx]
    elif token_mode == "rare_target":
        # Fewer a/b/digit tokens; tests counting robustness under sparse positives.
        values = torch.tensor([1, 2, 5, 6, 7, 8, 3, 3, 4, 4, 9, 9, 10, 10], dtype=torch.long)
        idx = torch.randint(0, len(values), (batch_size, max_length), generator=gen, dtype=torch.long)
        ids = values[idx]
    else:
        raise ValueError(f"unknown token_mode={token_mode}")

    pos = torch.arange(max_length, dtype=torch.long).unsqueeze(0)
    mask = pos < lengths.unsqueeze(1)
    ids = ids.masked_fill(~mask, PAD)

    task_ids_from_mix = torch.tensor([TASKS[t] for t in task_mix], dtype=torch.long)
    sampled = torch.randint(0, len(task_mix), (batch_size,), generator=gen, dtype=torch.long)
    task_ids = task_ids_from_mix[sampled]

    labels_length = lengths
    labels_count_a = ((ids == TOKENS["a"]) & mask).sum(dim=1)
    labels_count_b = ((ids == TOKENS["b"]) & mask).sum(dim=1)
    labels_count_digit = (((ids >= TOKENS["0"]) & (ids <= TOKENS["3"])) & mask).sum(dim=1)

    labels = torch.zeros(batch_size, dtype=torch.long)
    labels = torch.where(task_ids == TASKS["length"], labels_length, labels)
    labels = torch.where(task_ids == TASKS["count_a"], labels_count_a, labels)
    labels = torch.where(task_ids == TASKS["count_b"], labels_count_b, labels)
    labels = torch.where(task_ids == TASKS["count_digit"], labels_count_digit, labels)
    if int(labels.max()) >= num_classes:
        raise RuntimeError("label out of class range; increase --num-classes")

    return Batch(
        input_ids=ids.to(dev),
        task_ids=task_ids.to(dev),
        labels=labels.to(dev),
        attention_mask=mask.to(dtype=torch.long).to(dev),
    )


def make_batch_fn(
    min_length: int,
    max_length: int,
    task_mix: Sequence[str],
    token_mode: str,
    num_classes: int,
) -> Callable[[int, torch.device | str, int], Batch]:
    def fn(batch_size: int, device: torch.device | str, seed: int) -> Batch:
        return make_publication_batch(
            batch_size=batch_size,
            min_length=min_length,
            max_length=max_length,
            device=device,
            seed=seed,
            task_mix=task_mix,
            token_mode=token_mode,
            num_classes=num_classes,
        )
    return fn


def forward_model(model: torch.nn.Module, batch: Batch) -> torch.Tensor:
    return model(batch.input_ids, batch.task_ids, batch.attention_mask)


def eval_accuracy_and_loss(
    model: torch.nn.Module,
    batch_fn: Callable[[int, torch.device | str, int], Batch],
    batch_size: int,
    device: torch.device | str,
    seed_base: int,
    batches: int,
) -> Tuple[float, float]:
    model.eval()
    accs: List[float] = []
    losses: List[float] = []
    with torch.no_grad():
        for i in range(batches):
            batch = batch_fn(batch_size, device, seed_base + i)
            logits = forward_model(model, batch)
            accs.append(float(accuracy(logits, batch.labels)))
            losses.append(float(F.cross_entropy(logits, batch.labels).detach().cpu()))
    return float(np.mean(accs)), float(np.mean(losses))


def train_one(
    model_name: str,
    model_factory: Callable[[], torch.nn.Module],
    train_batch_fn: Callable[[int, torch.device | str, int], Batch],
    val_batch_fn: Callable[[int, torch.device | str, int], Batch],
    test_suites: Dict[str, Callable[[int, torch.device | str, int], Batch]],
    train_config: TrainConfig,
    seed: int,
    eval_batches: int,
) -> Dict[str, Any]:
    seed_everything(seed)
    device = train_config.device if train_config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory().to(device)
    model.train()

    objective = HtSB12Objective(
        margin=0.6,
        margin_weight=0.02,
        ratio_reg=1e-3,
        warmup_steps=train_config.warmup_steps,
        label_smoothing=0.1,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    best_val = -1.0
    best_step = 0
    best_state = None
    prev_loss = None
    spikes = 0
    max_spike = 0.0
    final_loss = float("nan")
    train_log: List[Dict[str, Any]] = []

    for step in range(1, train_config.steps + 1):
        lr = cosine_with_warmup(step, train_config.steps, train_config.warmup_steps, train_config.lr)
        for g in optim.param_groups:
            g["lr"] = lr

        batch = train_batch_fn(train_config.batch_size, device, seed * 1_000_000 + step)
        logits = forward_model(model, batch)
        loss_bd = objective(model, logits, batch.labels, step=step)
        loss = loss_bd.loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if train_config.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optim.step()

        final_loss = float(loss_bd.scalars()["loss"])
        if prev_loss is not None and prev_loss > 1e-8:
            ratio = final_loss / prev_loss
            if ratio > 1.5:
                spikes += 1
                max_spike = max(max_spike, ratio)
        prev_loss = final_loss

        if step % train_config.eval_every == 0 or step == train_config.steps:
            val_acc, val_loss = eval_accuracy_and_loss(
                model, val_batch_fn, min(512, train_config.batch_size * 2), device, seed * 2_000_000 + step * 100, eval_batches
            )
            if val_acc > best_val:
                best_val = val_acc
                best_step = step
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            row = {
                "model": model_name,
                "seed": seed,
                "step": step,
                "train_loss": round(final_loss, 4),
                "val_acc": round(val_acc * 100, 3),
                "val_ce_loss": round(val_loss, 4),
                "lr": lr,
            }
            if hasattr(model, "hts_diagnostics"):
                diag = model.hts_diagnostics()
                for k in ("l0_layer0_b12_delta_base_ratio", "l0_layer0_b12_gate_main", "l0_layer0_b12_gate_corr"):
                    if k in diag:
                        row[k] = round(diag[k], 5)
            train_log.append(row)
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    suite_results: Dict[str, Any] = {}
    for suite_name, suite_fn in test_suites.items():
        acc, ce_loss = eval_accuracy_and_loss(
            model, suite_fn, min(1024, max(512, train_config.batch_size * 2)), device, seed * 3_000_000 + len(suite_name) * 1000, eval_batches
        )
        suite_results[f"{suite_name}_acc"] = acc
        suite_results[f"{suite_name}_ce_loss"] = ce_loss

    return {
        "model": model_name,
        "seed": seed,
        "params": count_parameters(model),
        "best_val": best_val,
        "best_step": best_step,
        "spikes": spikes,
        "max_spike": max_spike,
        "final_loss": final_loss,
        **suite_results,
        "train_log": train_log,
    }


def eval_digit_accuracy_and_loss(
    model: torch.nn.Module,
    batch_fn: Callable[[int, torch.device | str, int], Batch],
    batch_size: int,
    device: torch.device | str,
    seed_base: int,
    batches: int,
) -> Tuple[float, float]:
    model.eval()
    accs: List[float] = []
    with torch.no_grad():
        for i in range(batches):
            batch = batch_fn(batch_size, device, seed_base + i)
            outputs = model(batch.input_ids, batch.task_ids, batch.attention_mask)
            accs.append(float(model.digit_accuracy(outputs, batch.labels)))
    return float(np.mean(accs)), 0.0


def train_one_digit(
    model_name: str,
    model_factory: Callable[[], torch.nn.Module],
    train_batch_fn: Callable[[int, torch.device | str, int], Batch],
    val_batch_fn: Callable[[int, torch.device | str, int], Batch],
    test_suites: Dict[str, Callable[[int, torch.device | str, int], Batch]],
    train_config: TrainConfig,
    seed: int,
    eval_batches: int,
) -> Dict[str, Any]:
    seed_everything(seed)
    device = train_config.device if train_config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory().to(device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    best_val = -1.0
    best_step = 0
    best_state = None
    prev_loss = None
    spikes = 0
    max_spike = 0.0
    final_loss = float("nan")
    train_log: List[Dict[str, Any]] = []

    for step in range(1, train_config.steps + 1):
        lr = cosine_with_warmup(step, train_config.steps, train_config.warmup_steps, train_config.lr)
        for g in optim.param_groups:
            g["lr"] = lr

        batch = train_batch_fn(train_config.batch_size, device, seed * 1_000_000 + step)
        outputs = model(batch.input_ids, batch.task_ids, batch.attention_mask)
        loss = model.digit_loss(outputs, batch.labels, label_smoothing=0.1, digit_weights=(0.1, 0.4, 0.5))
        if hasattr(model, "hts_regularizers"):
            budget, binary, ratio_penalty, task_offset_l2 = model.hts_regularizers()
            loss = loss + 1e-3 * budget + 1e-3 * binary + 0.01 * ratio_penalty + 1e-3 * task_offset_l2
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if train_config.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optim.step()

        final_loss = float(loss.detach().cpu())
        if prev_loss is not None and prev_loss > 1e-8:
            ratio = final_loss / prev_loss
            if ratio > 1.5:
                spikes += 1
                max_spike = max(max_spike, ratio)
        prev_loss = final_loss

        if step % train_config.eval_every == 0 or step == train_config.steps:
            val_acc, val_loss = eval_digit_accuracy_and_loss(
                model, val_batch_fn, min(512, train_config.batch_size * 2), device, seed * 2_000_000 + step * 100, eval_batches
            )
            if val_acc > best_val:
                best_val = val_acc
                best_step = step
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            row = {
                "model": model_name,
                "seed": seed,
                "step": step,
                "train_loss": round(final_loss, 4),
                "val_acc": round(val_acc * 100, 3),
                "val_ce_loss": round(val_loss, 4),
                "lr": lr,
            }
            if hasattr(model, "hts_diagnostics"):
                diag = model.hts_diagnostics()
                for k in ("l0_layer0_b12_delta_base_ratio", "l0_layer0_b12_gate_main", "l0_layer0_b12_gate_corr"):
                    if k in diag:
                        row[k] = round(diag[k], 5)
            train_log.append(row)
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    suite_results: Dict[str, Any] = {}
    for suite_name, suite_fn in test_suites.items():
        acc, ce_loss = eval_digit_accuracy_and_loss(
            model, suite_fn, min(1024, max(512, train_config.batch_size * 2)), device, seed * 3_000_000 + len(suite_name) * 1000, eval_batches
        )
        suite_results[f"{suite_name}_acc"] = acc
        suite_results[f"{suite_name}_ce_loss"] = ce_loss

    return {
        "model": model_name,
        "seed": seed,
        "params": count_parameters(model),
        "best_val": best_val,
        "best_step": best_step,
        "spikes": spikes,
        "max_spike": max_spike,
        "final_loss": final_loss,
        **suite_results,
        "train_log": train_log,
    }


def mean_std(values: Sequence[float]) -> Tuple[float, float, float, float]:
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.min()), float(arr.max())


def summarize_runs(runs: Sequence[Dict[str, Any]], suites: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    models = sorted(set(r["model"] for r in runs))
    for m in models:
        mr = [r for r in runs if r["model"] == m]
        row: Dict[str, Any] = {
            "model": m,
            "params": mr[0]["params"],
            "n_seeds": len(mr),
            "val_mean": round(mean_std([r["best_val"] for r in mr])[0] * 100, 3),
            "val_std": round(mean_std([r["best_val"] for r in mr])[1] * 100, 3),
            "avg_spikes": round(float(np.mean([r["spikes"] for r in mr])), 3),
            "max_spike": round(float(max(r["max_spike"] for r in mr)), 3),
        }
        for s in suites:
            vals = [r[f"{s}_acc"] for r in mr]
            losses = [r[f"{s}_ce_loss"] for r in mr]
            mu, sd, mn, mx = mean_std(vals)
            lmu, lsd, _, _ = mean_std(losses)
            row[f"{s}_acc_mean"] = round(mu * 100, 3)
            row[f"{s}_acc_std"] = round(sd * 100, 3)
            row[f"{s}_acc_min"] = round(mn * 100, 3)
            row[f"{s}_acc_max"] = round(mx * 100, 3)
            row[f"{s}_ce_mean"] = round(lmu, 5)
            row[f"{s}_ce_std"] = round(lsd, 5)
        rows.append(row)
    return rows


def build_hts_config(max_model_length: int, num_classes: int, task_mix: tuple = ("length", "count_a", "count_b", "count_digit")) -> HtSB12Config:
    num_tasks = len(task_mix)  # Auto-calculate from task mix
    return HtSB12Config(
        vocab_size=128,
        max_length=max_model_length,
        num_tasks=num_tasks,
        num_classes=num_classes,
        d_model=48,
        n_heads=4,
        num_layers=2,
        dim_ff=64,
        task_dim=12,
        rank_main=8,
        rank_corr=4,
        rank_task_attn=4,
        dropout=0.0,
        use_cls_token=False,
        pool="mean",
        alpha_max=1.05,
        target_min=0.20,
        target_max=0.90,
        corr_alpha_max=0.55,
        corr_gain=6.0,
        task_offset_scale=0.30,
        ratio_ceiling=2.0,
        corr_ceiling=1.0,
        router_per_task=True,
        use_pos_mod_basis=False,
        use_task_in_basis=True,
        use_mean_basis=False,
        use_ctx_basis=False,
        use_dual_delta=False,
        use_std_basis=False,
        label_smoothing=0.1,
        use_rms_norm=True,
        use_alibi=True,
    )


def build_transformer_config(kind: str, max_model_length: int, num_classes: int) -> HtSB12Config:
    if kind == "small":
        return HtSB12Config(
            vocab_size=128, max_length=max_model_length, num_tasks=8, num_classes=num_classes,
            d_model=80, n_heads=4, num_layers=2, dim_ff=128, dropout=0.0, use_cls_token=True, pool="cls",
        )
    if kind == "param_matched":
        # For num_classes=256 and max_model_length=200, this is within ~1% of HtS-B12 params.
        return HtSB12Config(
            vocab_size=128, max_length=max_model_length, num_tasks=8, num_classes=num_classes,
            d_model=80, n_heads=4, num_layers=3, dim_ff=128, dropout=0.0, use_cls_token=True, pool="cls",
        )
    raise KeyError(kind)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys and k != "train_log":
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def write_result_card(path: Path, summary_rows: Sequence[Dict[str, Any]], suites: Sequence[str], args: argparse.Namespace) -> None:
    by_model = {r["model"]: r for r in summary_rows}
    lines: List[str] = []
    lines.append("# HtS-B12 Publication Benchmark-B Result Card")
    lines.append("")
    lines.append("## Protocol")
    lines.append(f"- Seeds: `{args.seeds}`")
    lines.append(f"- Steps: `{args.steps}`")
    lines.append(f"- Train length: `1-{args.train_max_length}`")
    lines.append(f"- Held-out length: `{args.train_max_length + 1}-{args.max_eval_length}`")
    lines.append(f"- num_classes: `{args.num_classes}` (no label clipping)")
    lines.append(f"- Models: `{', '.join(by_model.keys())}`")
    lines.append("")
    lines.append("## Summary")
    header = ["Model", "Params"] + [f"{s} Acc mean±std" for s in suites] + ["Avg spikes"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in summary_rows:
        row = [r["model"], f"{r['params']:,}"]
        for s in suites:
            row.append(f"{r[f'{s}_acc_mean']:.2f} ± {r[f'{s}_acc_std']:.2f}%")
        row.append(str(r["avg_spikes"]))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Interpretation")
    if "HtS-B12" in by_model and "Transformer-ParamMatched" in by_model:
        h = by_model["HtS-B12"]
        t = by_model["Transformer-ParamMatched"]
        id_delta = h["id_acc_mean"] - t["id_acc_mean"]
        held_delta = h["heldout_length_acc_mean"] - t["heldout_length_acc_mean"]
        lines.append(f"- ID delta HtS vs param-matched Transformer: `{id_delta:+.2f} pp`.")
        lines.append(f"- Held-out length delta HtS vs param-matched Transformer: `{held_delta:+.2f} pp`.")
        if id_delta > 1.0 and held_delta > 0.0:
            lines.append("- Claim level: positive preliminary evidence across ID and held-out length.")
        elif id_delta > 1.0:
            lines.append("- Claim level: positive ID evidence, but held-out length still needs caution.")
        else:
            lines.append("- Claim level: no clear accuracy win over param-matched Transformer.")
    if "HtS-NoSoft" in by_model and "HtS-B12" in by_model:
        h = by_model["HtS-B12"]
        n = by_model["HtS-NoSoft"]
        lines.append(f"- Soft-update ablation delta on ID: `{h['id_acc_mean'] - n['id_acc_mean']:+.2f} pp`.")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This benchmark intentionally uses a parameter-matched Transformer baseline and avoids silent label clipping.")
    lines.append("- For a definitive paper claim, report this card together with hardware efficiency: throughput, latency, peak VRAM, and FLOPs/MACs.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Very fast smoke run")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--train-max-length", type=int, default=128)
    parser.add_argument("--max-eval-length", type=int, default=200)
    parser.add_argument("--num-classes", type=int, default=256)
    parser.add_argument("--include-small-transformer", action="store_true", help="Also run smaller Transformer baseline")
    parser.add_argument("--include-ablation", action="store_true", help="Run HtS-NoSoft ablation; slower")
    parser.add_argument("--use-digit-head", action="store_true", help="Use digit decomposition head instead of standard classifier")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    if args.quick:
        args.steps = min(args.steps, 300)
        args.batch_size = min(args.batch_size, 96)
        args.eval_batches = 1
        args.seeds = "42"

    if args.num_classes <= args.max_eval_length:
        raise SystemExit("For publication benchmark, --num-classes must exceed --max-eval-length to avoid label clipping.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    print(f"Seeds: {seeds}")
    print(f"Train length: 1-{args.train_max_length}; held-out length: {args.train_max_length+1}-{args.max_eval_length}")
    print("No label clipping: num_classes > max_eval_length verified.")

    hts_cfg = build_hts_config(args.max_eval_length, args.num_classes)
    tf_small_cfg = build_transformer_config("small", args.max_eval_length, args.num_classes)
    tf_pm_cfg = build_transformer_config("param_matched", args.max_eval_length, args.num_classes)

    hts_models: List[Tuple[str, Callable[[], torch.nn.Module]]]
    if args.use_digit_head:
        hts_models = [("HtS-B12-Digit", lambda: HtSB12DigitClassifier(hts_cfg, max_digit_value=args.num_classes - 1))]
    else:
        hts_models = [("HtS-B12", lambda: HtSB12Classifier(hts_cfg))]

    transformer_models: List[Tuple[str, Callable[[], torch.nn.Module]]] = [
        ("Transformer-ParamMatched", lambda: TransformerClassifier(tf_pm_cfg)),
    ]
    if args.include_small_transformer:
        transformer_models.append(("Transformer-Small", lambda: TransformerClassifier(tf_small_cfg)))
    if args.include_ablation:
        def no_soft_factory() -> torch.nn.Module:
            cfg = copy.deepcopy(hts_cfg)
            cfg.alpha_max = 0.0
            cfg.corr_alpha_max = 0.0
            cfg.corr_gain = 0.0
            return HtSB12Classifier(cfg)
        transformer_models.append(("HtS-NoSoft", no_soft_factory))

    models = hts_models + transformer_models

    print("\nParameter counts:")
    for name, factory in models:
        p = count_parameters(factory())
        print(f"  {name:<26} {p:,}")

    task_full = ("length", "count_a", "count_b", "count_digit")
    task_length = ("length",)
    task_count = ("count_a", "count_b", "count_digit")

    train_fn = make_batch_fn(1, args.train_max_length, task_full, "uniform", args.num_classes)
    val_fn = make_batch_fn(1, args.train_max_length, task_full, "uniform", args.num_classes)
    suites = {
        "id": make_batch_fn(1, args.train_max_length, task_full, "uniform", args.num_classes),
        "heldout_length": make_batch_fn(args.train_max_length + 1, args.max_eval_length, task_full, "uniform", args.num_classes),
        "length_only": make_batch_fn(1, args.max_eval_length, task_length, "uniform", args.num_classes),
        "count_only": make_batch_fn(1, args.max_eval_length, task_count, "uniform", args.num_classes),
        "biased_count": make_batch_fn(1, args.max_eval_length, task_count, "biased_count", args.num_classes),
    }
    suite_names = list(suites.keys())

    tc = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=1e-3,
        weight_decay=0.05,
        warmup_steps=max(20, min(250, args.steps // 20)),
        grad_clip=1.0,
        eval_every=max(50, args.steps // 10),
        device=device,
        seed=42,
    )

    all_runs: List[Dict[str, Any]] = []
    t0 = time.time()
    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"Seed {seed}")
        print("=" * 80)
        for model_name, factory in models:
            print(f"\nTraining {model_name}...")
            if args.use_digit_head and "Digit" in model_name:
                r = train_one_digit(
                    model_name,
                    factory,
                    train_fn,
                    val_fn,
                    suites,
                    tc,
                    seed,
                    args.eval_batches,
                )
            else:
                r = train_one(
                    model_name,
                    factory,
                    train_fn,
                    val_fn,
                    suites,
                    tc,
                    seed,
                    args.eval_batches,
                )
            all_runs.append(r)
            msg = (
                f"  {model_name:<26} val={r['best_val']*100:6.2f}% "
                f"id={r['id_acc']*100:6.2f}% heldout={r['heldout_length_acc']*100:6.2f}% "
                f"best_step={r['best_step']} spikes={r['spikes']} max_spike={r['max_spike']:.2f}x"
            )
            print(msg)

    summary = summarize_runs(all_runs, suite_names)
    print("\n" + "=" * 80)
    print("PUBLICATION BENCHMARK-B SUMMARY")
    print("=" * 80)
    for r in summary:
        print(f"\n{r['model']}  params={r['params']:,} seeds={r['n_seeds']}")
        for s in suite_names:
            print(f"  {s:<16} {r[f'{s}_acc_mean']:6.2f} ± {r[f'{s}_acc_std']:.2f}%  CE={r[f'{s}_ce_mean']:.4f}")
        print(f"  avg_spikes={r['avg_spikes']} max_spike={r['max_spike']}")

    flat_runs: List[Dict[str, Any]] = []
    for r in all_runs:
        rr = {k: v for k, v in r.items() if k != "train_log"}
        flat_runs.append(rr)

    write_csv(out_dir / "all_runs.csv", flat_runs)
    write_csv(out_dir / "summary.csv", summary)
    # training curves are separated to keep all_runs readable.
    train_rows: List[Dict[str, Any]] = []
    for r in all_runs:
        train_rows.extend(r["train_log"])
    write_csv(out_dir / "training_curves.csv", train_rows)

    config_card = {
        "seeds": seeds,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_max_length": args.train_max_length,
        "max_eval_length": args.max_eval_length,
        "num_classes": args.num_classes,
        "device": device,
        "models": [name for name, _ in models],
    }
    (out_dir / "config.json").write_text(json.dumps(config_card, indent=2), encoding="utf-8")
    write_result_card(out_dir / "result_card.md", summary, suite_names, args)
    print(f"\nSaved results to: {out_dir.resolve()}")
    print(f"Elapsed: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
