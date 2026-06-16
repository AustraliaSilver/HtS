"""Publication Benchmark-C: diagnostic generalization benchmark for HtS-B12.

Benchmark-B showed a clear in-distribution advantage for HtS-B12 but poor
held-out length extrapolation.  Benchmark-C is designed to identify *why*.
It separates three questions:

C1. Seen full range: if the model is trained on lengths 1..200, can it solve
    the full range?  This tests basic capacity / label availability.
C2. Bucket interpolation: if positions up to 200 are seen but some length
    buckets are withheld, can the model generalize to withheld buckets?  This
    reduces pure positional extrapolation and probes interpolation/generalization.
C3. Length extrapolation: train 1..128 and test 129..200, matching the hard
    extrapolation condition from Benchmark-B.

Recommended Kaggle commands
---------------------------
Quick smoke:
    !python /kaggle/working/HtS/benchmarks/kaggle_publication_c.py --quick

Run one protocol first (recommended):
    !python /kaggle/working/HtS/benchmarks/kaggle_publication_c.py \
        --protocols c1 --steps 5000 --seeds 42,123,777

Full diagnostic run, slower:
    !python /kaggle/working/HtS/benchmarks/kaggle_publication_c.py \
        --protocols c1,c2,c3 --steps 5000 --seeds 42,123,777

Outputs:
    publication_c_results/summary.csv
    publication_c_results/all_runs.csv
    publication_c_results/training_curves.csv
    publication_c_results/result_card.md
    publication_c_results/config.json
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Prefer local source over stale installed package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from hts_b12 import HtSB12Classifier, HtSB12Config, TransformerClassifier, TrainConfig, accuracy, count_parameters
from hts_b12.losses import HtSB12Objective
from hts_b12.training import cosine_with_warmup

DEFAULT_SEEDS = [42, 123, 777]
RESULTS_DIR = Path("publication_c_results")

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


def _sample_lengths(allowed_lengths: Sequence[int], batch_size: int, gen: torch.Generator | None) -> torch.Tensor:
    if not allowed_lengths:
        raise ValueError("allowed_lengths is empty")
    allowed = torch.tensor(list(allowed_lengths), dtype=torch.long)
    idx = torch.randint(0, len(allowed), (batch_size,), generator=gen, dtype=torch.long)
    return allowed[idx]


def make_batch(
    batch_size: int,
    allowed_lengths: Sequence[int],
    device: torch.device | str,
    seed: int | None,
    task_mix: Sequence[str],
    token_mode: str = "uniform",
    num_classes: int = 256,
) -> Batch:
    """Vectorized synthetic batch with explicit allowed length set.

    No label clipping is allowed. If a label exceeds the class range, the function
    raises to avoid contaminating extrapolation claims.
    """
    for t in task_mix:
        if t not in TASKS:
            raise KeyError(f"unknown task {t!r}; allowed={list(TASKS)}")
    gen = torch.Generator(device="cpu") if seed is not None else None
    if gen is not None:
        gen.manual_seed(int(seed))
    dev = torch.device(device)

    lengths = _sample_lengths(allowed_lengths, batch_size, gen)
    max_len = int(max(allowed_lengths))
    if max_len >= num_classes:
        raise ValueError(f"num_classes={num_classes} must exceed max length {max_len}")

    if token_mode == "uniform":
        ids = torch.randint(1, len(TOKENS) + 1, (batch_size, max_len), generator=gen, dtype=torch.long)
    elif token_mode == "biased_count":
        # Positive targets more frequent. Tests count robustness under distribution shift.
        values = torch.tensor([1, 1, 1, 2, 2, 5, 6, 7, 8, 9, 10, 3, 4], dtype=torch.long)
        idx = torch.randint(0, len(values), (batch_size, max_len), generator=gen, dtype=torch.long)
        ids = values[idx]
    elif token_mode == "rare_target":
        # Positive targets rarer. Tests sparse counting robustness.
        values = torch.tensor([1, 2, 5, 6, 7, 8, 3, 3, 4, 4, 9, 9, 10, 10], dtype=torch.long)
        idx = torch.randint(0, len(values), (batch_size, max_len), generator=gen, dtype=torch.long)
        ids = values[idx]
    else:
        raise ValueError(f"unknown token_mode={token_mode}")

    pos = torch.arange(max_len, dtype=torch.long).unsqueeze(0)
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


def batch_fn(
    allowed_lengths: Sequence[int],
    task_mix: Sequence[str],
    token_mode: str,
    num_classes: int,
) -> Callable[[int, torch.device | str, int], Batch]:
    def fn(batch_size: int, device: torch.device | str, seed: int) -> Batch:
        return make_batch(batch_size, allowed_lengths, device, seed, task_mix, token_mode, num_classes)
    return fn


def forward_model(model: torch.nn.Module, batch: Batch) -> torch.Tensor:
    return model(batch.input_ids, batch.task_ids, batch.attention_mask)


def eval_accuracy_and_loss(
    model: torch.nn.Module,
    fn: Callable[[int, torch.device | str, int], Batch],
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
            b = fn(batch_size, device, seed_base + i)
            logits = forward_model(model, b)
            accs.append(float(accuracy(logits, b.labels)))
            losses.append(float(F.cross_entropy(logits, b.labels).detach().cpu()))
    return float(np.mean(accs)), float(np.mean(losses))


def train_one(
    protocol: str,
    model_name: str,
    model_factory: Callable[[], torch.nn.Module],
    train_fn: Callable[[int, torch.device | str, int], Batch],
    val_fn: Callable[[int, torch.device | str, int], Batch],
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
        margin_weight=0.03,
        ratio_reg=1e-3,
        warmup_steps=train_config.warmup_steps,
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

        batch = train_fn(train_config.batch_size, device, seed * 1_000_000 + step)
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
                model, val_fn, min(512, train_config.batch_size * 2), device, seed * 2_000_000 + step * 100, eval_batches
            )
            if val_acc > best_val:
                best_val = val_acc
                best_step = step
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            row: Dict[str, Any] = {
                "protocol": protocol,
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
                        row[k] = round(float(diag[k]), 5)
            train_log.append(row)
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    suite_results: Dict[str, Any] = {}
    for suite_name, suite_fn in test_suites.items():
        # Full runs use a large evaluation batch; tiny smoke runs avoid a CPU bottleneck.
        eval_bs = train_config.batch_size * 2 if train_config.steps <= 5 else min(1024, max(512, train_config.batch_size * 2))
        acc, ce_loss = eval_accuracy_and_loss(
            model, suite_fn, eval_bs, device, seed * 3_000_000 + len(suite_name) * 1000, eval_batches
        )
        suite_results[f"{suite_name}_acc"] = acc
        suite_results[f"{suite_name}_ce_loss"] = ce_loss

    return {
        "protocol": protocol,
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


def summarize_runs(runs: Sequence[Dict[str, Any]], suite_names_by_protocol: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    protocols = sorted(set(r["protocol"] for r in runs))
    for p in protocols:
        suites = suite_names_by_protocol[p]
        pr = [r for r in runs if r["protocol"] == p]
        for m in sorted(set(r["model"] for r in pr)):
            mr = [r for r in pr if r["model"] == m]
            row: Dict[str, Any] = {
                "protocol": p,
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


def build_hts_config(max_model_length: int, num_classes: int) -> HtSB12Config:
    return HtSB12Config(
        vocab_size=128,
        max_length=max_model_length,
        num_tasks=8,
        num_classes=num_classes,
        d_model=80,
        n_heads=4,
        num_layers=2,
        dim_ff=128,
        task_dim=16,
        rank_main=8,
        rank_corr=4,
        dropout=0.0,
        use_cls_token=True,
        pool="cls",
        alpha_max=1.05,
        target_min=0.20,
        target_max=0.75,
        corr_alpha_max=0.35,
        corr_gain=3.0,
        task_offset_scale=0.20,
        ratio_ceiling=0.80,
        corr_ceiling=0.25,
    )


def build_transformer_config(max_model_length: int, num_classes: int) -> HtSB12Config:
    # Within ~1% of HtS-B12 for num_classes=256 and max_length=200.
    return HtSB12Config(
        vocab_size=128,
        max_length=max_model_length,
        num_tasks=8,
        num_classes=num_classes,
        d_model=80,
        n_heads=4,
        num_layers=3,
        dim_ff=128,
        dropout=0.0,
        use_cls_token=True,
        pool="cls",
    )


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


def protocol_defs(max_eval_length: int, train_max_length: int, holdout_mod: int, num_classes: int) -> Dict[str, Dict[str, Any]]:
    full_tasks = ("length", "count_a", "count_b", "count_digit")
    length_task = ("length",)
    count_tasks = ("count_a", "count_b", "count_digit")
    all_200 = list(range(1, max_eval_length + 1))
    extrap_train = list(range(1, train_max_length + 1))
    extrap_test = list(range(train_max_length + 1, max_eval_length + 1))
    bucket_holdout = [x for x in all_200 if x % holdout_mod == 0]
    bucket_train = [x for x in all_200 if x % holdout_mod != 0]

    return {
        "c1_seen_full_range": {
            "description": "Train and test on lengths 1..max_eval_length; tests basic capacity and label availability.",
            "train": batch_fn(all_200, full_tasks, "uniform", num_classes),
            "val": batch_fn(all_200, full_tasks, "uniform", num_classes),
            "suites": {
                "full_id": batch_fn(all_200, full_tasks, "uniform", num_classes),
                "full_length_only": batch_fn(all_200, length_task, "uniform", num_classes),
                "full_count_only": batch_fn(all_200, count_tasks, "uniform", num_classes),
                "full_biased_count": batch_fn(all_200, count_tasks, "biased_count", num_classes),
                "full_rare_count": batch_fn(all_200, count_tasks, "rare_target", num_classes),
            },
        },
        "c2_bucket_interpolation": {
            "description": f"Train on 1..{max_eval_length} except lengths divisible by {holdout_mod}; test withheld buckets.",
            "train": batch_fn(bucket_train, full_tasks, "uniform", num_classes),
            "val": batch_fn(bucket_train, full_tasks, "uniform", num_classes),
            "suites": {
                "seen_buckets": batch_fn(bucket_train, full_tasks, "uniform", num_classes),
                "heldout_buckets": batch_fn(bucket_holdout, full_tasks, "uniform", num_classes),
                "heldout_length_only": batch_fn(bucket_holdout, length_task, "uniform", num_classes),
                "heldout_count_only": batch_fn(bucket_holdout, count_tasks, "uniform", num_classes),
            },
        },
        "c3_extrapolation": {
            "description": f"Train on 1..{train_max_length}; test on {train_max_length+1}..{max_eval_length}.",
            "train": batch_fn(extrap_train, full_tasks, "uniform", num_classes),
            "val": batch_fn(extrap_train, full_tasks, "uniform", num_classes),
            "suites": {
                "id_1_trainmax": batch_fn(extrap_train, full_tasks, "uniform", num_classes),
                "extrap_length": batch_fn(extrap_test, full_tasks, "uniform", num_classes),
                "extrap_length_only": batch_fn(extrap_test, length_task, "uniform", num_classes),
                "extrap_count_only": batch_fn(extrap_test, count_tasks, "uniform", num_classes),
                "extrap_biased_count": batch_fn(extrap_test, count_tasks, "biased_count", num_classes),
            },
        },
    }


def write_result_card(path: Path, summary: Sequence[Dict[str, Any]], proto_meta: Dict[str, Dict[str, Any]], args: argparse.Namespace) -> None:
    lines: List[str] = []
    lines.append("# HtS-B12 Publication Benchmark-C Diagnostic Result Card")
    lines.append("")
    lines.append("## Protocol")
    lines.append(f"- Seeds: `{args.seeds}`")
    lines.append(f"- Steps: `{args.steps}`")
    lines.append(f"- Batch size: `{args.batch_size}`")
    lines.append(f"- Max eval length: `{args.max_eval_length}`")
    lines.append(f"- Train max length for extrapolation protocol: `{args.train_max_length}`")
    lines.append(f"- num_classes: `{args.num_classes}` (must exceed max_eval_length; no label clipping)")
    lines.append(f"- Protocols run: `{args.protocols}`")
    lines.append("")
    lines.append("## Protocol descriptions")
    for p, meta in proto_meta.items():
        lines.append(f"- `{p}`: {meta['description']}")
    lines.append("")
    lines.append("## Summary")
    for p in sorted(set(r["protocol"] for r in summary)):
        lines.append(f"\n### {p}")
        rows = [r for r in summary if r["protocol"] == p]
        # Collect suite keys dynamically.
        suite_prefixes = []
        for k in rows[0].keys():
            if k.endswith("_acc_mean"):
                suite_prefixes.append(k[:-9])
        header = ["Model", "Params", "Val"] + [f"{s} Acc" for s in suite_prefixes] + ["Avg spikes"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in rows:
            row = [r["model"], f"{r['params']:,}", f"{r['val_mean']:.2f} ± {r['val_std']:.2f}%"]
            for s in suite_prefixes:
                row.append(f"{r[f'{s}_acc_mean']:.2f} ± {r[f'{s}_acc_std']:.2f}%")
            row.append(str(r["avg_spikes"]))
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## How to interpret Benchmark-C")
    lines.append("- If C1 is high but C3 is low, the failure is mainly length extrapolation / unseen range rather than raw capacity.")
    lines.append("- If C2 is high but C3 is low, positions up to the max length are usable, but extrapolating beyond the trained length range fails.")
    lines.append("- If C2 is also low, the model may be memorizing length buckets or class boundaries rather than learning a length/count rule.")
    lines.append("- If biased/rare count suites are low, count robustness remains a separate weakness.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Very fast smoke run")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--protocols", type=str, default="c1,c2,c3", help="Comma list from c1,c2,c3")
    parser.add_argument("--train-max-length", type=int, default=128)
    parser.add_argument("--max-eval-length", type=int, default=200)
    parser.add_argument("--num-classes", type=int, default=256)
    parser.add_argument("--holdout-mod", type=int, default=7)
    parser.add_argument("--include-ablation", action="store_true", help="Also run HtS-NoSoft; slower")
    parser.add_argument(
        "--model-filter",
        type=str,
        default="all",
        help="Comma list: all, hts-b12, transformer, no-soft. Useful for dual-GPU launchers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for this process: auto, cpu, cuda, cuda:0, etc. With CUDA_VISIBLE_DEVICES set, use cuda.",
    )
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    if args.quick:
        # Keep the smoke test tiny enough to run even on CPU. Full Kaggle runs
        # should omit --quick and use max_eval_length=200.
        args.steps = min(args.steps, 1)
        args.batch_size = min(args.batch_size, 2)
        args.eval_batches = 1
        args.seeds = "42"
        args.protocols = "c1"
        args.max_eval_length = min(args.max_eval_length, 16)
        args.train_max_length = min(args.train_max_length, 10)
        args.num_classes = max(args.num_classes, 128)

    if args.num_classes <= args.max_eval_length:
        raise SystemExit("--num-classes must exceed --max-eval-length to avoid label clipping.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"Requested {device}, but CUDA is not available.")
    print(f"Device: {device}")
    if device.startswith("cuda"):
        local_idx = 0
        if ":" in device:
            try:
                local_idx = int(device.split(":", 1)[1])
            except ValueError:
                local_idx = 0
        visible = __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "<not-set>")
        print(f"CUDA_VISIBLE_DEVICES: {visible}")
        print(f"GPU: {torch.cuda.get_device_name(local_idx)}")
        print(f"VRAM: {torch.cuda.get_device_properties(local_idx).total_memory / 1e9:.1f} GB")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    proto_map = {"c1": "c1_seen_full_range", "c2": "c2_bucket_interpolation", "c3": "c3_extrapolation"}
    requested = [p.strip().lower() for p in args.protocols.split(",") if p.strip()]
    protocols = [proto_map.get(p, p) for p in requested]

    hts_cfg = build_hts_config(args.max_eval_length, args.num_classes)
    tf_cfg = build_transformer_config(args.max_eval_length, args.num_classes)
    requested_models = {m.strip().lower() for m in args.model_filter.split(",") if m.strip()}
    if not requested_models or "all" in requested_models:
        requested_models = {"hts-b12", "transformer"}
        if args.include_ablation:
            requested_models.add("no-soft")

    models: List[Tuple[str, Callable[[], torch.nn.Module]]] = []
    if "hts-b12" in requested_models or "hts" in requested_models:
        models.append(("HtS-B12", lambda: HtSB12Classifier(hts_cfg)))
    if "transformer" in requested_models or "tfm" in requested_models or "transformer-parammatched" in requested_models:
        models.append(("Transformer-ParamMatched", lambda: TransformerClassifier(tf_cfg)))
    if "no-soft" in requested_models or "hts-nosoft" in requested_models or "nosoft" in requested_models:
        def no_soft_factory() -> torch.nn.Module:
            cfg = copy.deepcopy(hts_cfg)
            cfg.alpha_max = 0.0
            cfg.corr_alpha_max = 0.0
            cfg.corr_gain = 0.0
            return HtSB12Classifier(cfg)
        models.append(("HtS-NoSoft", no_soft_factory))
    if not models:
        raise SystemExit(f"No models selected by --model-filter={args.model_filter!r}")

    print(f"Seeds: {seeds}")
    print(f"Protocols: {protocols}")
    print(f"Max eval length: {args.max_eval_length}; num_classes: {args.num_classes} (no clipping)")
    print("\nParameter counts:")
    for name, factory in models:
        print(f"  {name:<26} {count_parameters(factory()):,}")

    all_proto = protocol_defs(args.max_eval_length, args.train_max_length, args.holdout_mod, args.num_classes)
    proto_meta = {p: all_proto[p] for p in protocols}
    suite_names_by_protocol = {p: list(meta["suites"].keys()) for p, meta in proto_meta.items()}

    tc = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=1e-3,
        weight_decay=0.01,
        warmup_steps=max(20, min(250, args.steps // 20)),
        grad_clip=1.0,
        eval_every=max(50, args.steps // 10),
        device=device,
        seed=42,
    )

    all_runs: List[Dict[str, Any]] = []
    t0 = time.time()
    for protocol in protocols:
        meta = proto_meta[protocol]
        print("\n" + "#" * 88)
        print(f"PROTOCOL {protocol}: {meta['description']}")
        print("#" * 88)
        for seed in seeds:
            print("\n" + "=" * 80)
            print(f"{protocol} | Seed {seed}")
            print("=" * 80)
            for model_name, factory in models:
                print(f"\nTraining {model_name}...")
                r = train_one(
                    protocol,
                    model_name,
                    factory,
                    meta["train"],
                    meta["val"],
                    meta["suites"],
                    tc,
                    seed,
                    args.eval_batches,
                )
                all_runs.append(r)
                first_suite = list(meta["suites"].keys())[0]
                msg = (
                    f"  {model_name:<26} val={r['best_val']*100:6.2f}% "
                    f"{first_suite}={r[first_suite + '_acc']*100:6.2f}% "
                    f"best_step={r['best_step']} spikes={r['spikes']} max_spike={r['max_spike']:.2f}x"
                )
                print(msg)

    summary = summarize_runs(all_runs, suite_names_by_protocol)
    print("\n" + "=" * 88)
    print("PUBLICATION BENCHMARK-C SUMMARY")
    print("=" * 88)
    for r in summary:
        print(f"\n[{r['protocol']}] {r['model']}  params={r['params']:,} seeds={r['n_seeds']}")
        for k in r:
            if k.endswith("_acc_mean"):
                s = k[:-9]
                print(f"  {s:<22} {r[k]:6.2f} ± {r[f'{s}_acc_std']:.2f}%  CE={r[f'{s}_ce_mean']:.4f}")
        print(f"  avg_spikes={r['avg_spikes']} max_spike={r['max_spike']}")

    flat_runs: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    for r in all_runs:
        flat_runs.append({k: v for k, v in r.items() if k != "train_log"})
        train_rows.extend(r["train_log"])
    write_csv(out_dir / "all_runs.csv", flat_runs)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "training_curves.csv", train_rows)
    config = {
        "seeds": seeds,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_batches": args.eval_batches,
        "protocols": protocols,
        "train_max_length": args.train_max_length,
        "max_eval_length": args.max_eval_length,
        "num_classes": args.num_classes,
        "holdout_mod": args.holdout_mod,
        "device": device,
        "model_filter": args.model_filter,
        "models": [m for m, _ in models],
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_result_card(out_dir / "result_card.md", summary, proto_meta, args)
    print(f"\nSaved results to: {out_dir.resolve()}")
    print(f"Elapsed: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
