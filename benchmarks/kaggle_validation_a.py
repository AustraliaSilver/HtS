"""HtS-B12 vs Transformer: Validation-A Benchmark (Kaggle T4/T4x2)

Fixed/stabilized version.

Main fixes compared with the first public Kaggle script:
1. Model initialization is now seeded correctly.  The old script constructed
   the model before calling seed_everything(seed), so the reported per-seed
   runs did not control initialization.
2. Batches pass attention_mask into both HtS and Transformer.
3. Test accuracy is measured from the best-validation checkpoint, not blindly
   from the final step.
4. Data generation in hts_b12.data.string_tasks is vectorized in this package.
5. A2 print bug is fixed: HtS and Transformer are no longer printed from the
   same overwritten variable.

Usage on Kaggle:
    !python /kaggle/working/HtS/benchmarks/kaggle_validation_a.py

Quick smoke:
    !python /kaggle/working/HtS/benchmarks/kaggle_validation_a.py --quick
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from hts_b12 import (
    HtSB12Classifier,
    HtSB12Config,
    TransformerClassifier,
    TrainConfig,
    accuracy,
    count_parameters,
)
from hts_b12.data.string_tasks import make_string_count_batch
from hts_b12.training import cosine_with_warmup
from hts_b12.losses import HtSB12Objective

DEFAULT_SEEDS = [42, 123, 777, 2024, 9999]
RESULTS_DIR = Path("validation_a_results")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def make_batch_fn(max_length: int, num_classes: int, task_mix: Tuple[str, ...], min_length: int = 1):
    def fn(batch_size: int, device: torch.device | str, seed: int):
        return make_string_count_batch(
            batch_size=batch_size,
            min_length=min_length,
            max_length=max_length,
            device=device,
            num_classes=num_classes,
            seed=seed,
            task_mix=task_mix,
        )
    return fn


def forward_model(model: torch.nn.Module, batch) -> torch.Tensor:
    return model(batch.input_ids, batch.task_ids, getattr(batch, "attention_mask", None))


def eval_accuracy(
    model: torch.nn.Module,
    batch_fn: Callable,
    batch_size: int,
    device: torch.device | str,
    seed_base: int,
    batches: int = 4,
) -> float:
    vals: List[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(batches):
            batch = batch_fn(batch_size, device, seed_base + i)
            logits = forward_model(model, batch)
            vals.append(float(accuracy(logits, batch.labels)))
    return float(np.mean(vals))


def train_one(
    model_name: str,
    model_factory: Callable[[], torch.nn.Module],
    train_batch_fn: Callable,
    eval_batch_fn: Callable,
    config: TrainConfig,
    seed: int,
    eval_batches: int = 4,
) -> Dict[str, Any]:
    seed_everything(seed)
    device = config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory().to(device)
    model.train()

    is_hts = isinstance(model, HtSB12Classifier)
    objective = HtSB12Objective(
        margin=0.6,
        margin_weight=0.03,
        ratio_reg=1e-3,
        warmup_steps=config.warmup_steps,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val = -1.0
    best_step = 0
    best_state = None
    prev_loss = None
    n_spikes = 0
    max_spike = 0.0
    log: List[Dict[str, Any]] = []
    cur_loss = float("nan")

    for step in range(1, config.steps + 1):
        lr = cosine_with_warmup(step, config.steps, config.warmup_steps, config.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        batch = train_batch_fn(config.batch_size, device, seed * 1_000_000 + step)
        logits = forward_model(model, batch)
        loss_bd = objective(model, logits, batch.labels, step=step)
        loss = loss_bd.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        cur_loss = float(loss_bd.scalars()["loss"])
        if prev_loss is not None and prev_loss > 1e-8:
            spike_ratio = cur_loss / prev_loss
            if spike_ratio > 1.5:
                n_spikes += 1
                max_spike = max(max_spike, spike_ratio)
        prev_loss = cur_loss

        if step % config.eval_every == 0 or step == config.steps:
            tr_acc = float(accuracy(logits, batch.labels))
            val_acc = eval_accuracy(
                model,
                eval_batch_fn,
                batch_size=min(512, config.batch_size * 2),
                device=device,
                seed_base=seed * 2_000_000 + step * 100,
                batches=eval_batches,
            )
            model.train()

            if val_acc > best_val:
                best_val = val_acc
                best_step = step
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

            row: Dict[str, Any] = {
                "step": step,
                "seed": seed,
                "loss": round(cur_loss, 4),
                "train_acc": round(tr_acc * 100, 2),
                "val_acc": round(val_acc * 100, 2),
                "lr": lr,
            }
            if is_hts:
                diag = model.hts_diagnostics()
                for k in (
                    "l0_layer0_b12_delta_base_ratio",
                    "l0_layer0_b12_gate_main",
                    "l0_layer0_b12_gate_corr",
                    "l1_layer1_b12_delta_base_ratio",
                    "l1_layer1_b12_gate_main",
                    "l1_layer1_b12_gate_corr",
                ):
                    if k in diag:
                        row[k] = round(diag[k], 4)
            log.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    test_acc = eval_accuracy(
        model,
        eval_batch_fn,
        batch_size=512,
        device=device,
        seed_base=seed * 3_000_000,
        batches=max(eval_batches, 4),
    )
    return {
        "model": model_name,
        "seed": seed,
        "params": count_parameters(model),
        "best_val": best_val,
        "best_step": best_step,
        "test": test_acc,
        "spikes": n_spikes,
        "max_spike": round(max_spike, 2),
        "final_loss": round(cur_loss, 4),
        "log": log,
    }


def agg(results: List[Dict[str, Any]]) -> Dict[str, float]:
    vals = [r["best_val"] for r in results]
    tests = [r["test"] for r in results]
    return {
        "val_mean": round(float(np.mean(vals)) * 100, 2),
        "val_std": round(float(np.std(vals)) * 100, 2),
        "test_mean": round(float(np.mean(tests)) * 100, 2),
        "test_std": round(float(np.std(tests)) * 100, 2),
        "avg_spikes": round(float(np.mean([r["spikes"] for r in results])), 1),
        "max_spike": round(float(max(r["max_spike"] for r in results)), 2),
    }


def make_config(max_length: int, num_classes: int) -> HtSB12Config:
    return HtSB12Config(
        vocab_size=128,
        max_length=max_length,
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


def print_run(prefix: str, r: Dict[str, Any]) -> None:
    print(
        f"    {prefix:<12} val={r['best_val']*100:5.1f}%  "
        f"test={r['test']*100:5.1f}%  best_step={r['best_step']:4d}  "
        f"spikes={r['spikes']}  max_spike={r['max_spike']}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2-only", action="store_true", help="Run only A2 held-out tests")
    parser.add_argument("--quick", action="store_true", help="Fast smoke run: 2 seeds, fewer steps")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list")
    parser.add_argument("--max-length", type=int, default=200, help="Maximum sequence length for the main benchmark")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    max_length = int(args.max_length)
    num_classes = 128
    cfg = make_config(max_length, num_classes)

    hts_params = count_parameters(HtSB12Classifier(cfg))
    tf_params = count_parameters(TransformerClassifier(cfg))
    print(f"HtS-B12 params:     {hts_params:,}")
    print(f"Transformer params: {tf_params:,}")
    print(f"Param diff:         {hts_params - tf_params:,} ({(hts_params - tf_params)/tf_params*100:+.1f}%)")
    print("NOTE: this script uses best-validation checkpoint and seeded initialization.")

    seeds = DEFAULT_SEEDS
    if args.seeds:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if args.quick:
        seeds = seeds[:2]

    steps = args.steps if args.steps is not None else (800 if args.quick else 5000)
    batch_size = args.batch_size if args.batch_size is not None else (128 if args.quick else 256)
    tc = TrainConfig(
        steps=steps,
        batch_size=batch_size,
        lr=1e-3,
        weight_decay=0.01,
        warmup_steps=max(20, min(250, steps // 20)),
        grad_clip=1.0,
        eval_every=max(50, steps // 10),
        device=device,
        seed=42,
    )
    full_tasks = ("length", "count_a", "count_b", "count_digit")
    t0 = time.time()
    all_summary: Dict[str, Any] = {}
    all_runs: List[Dict[str, Any]] = []

    if not args.a2_only:
        print("\n" + "=" * 70)
        print("  A1: STABILITY — seeded init, cosine warmup, spike detection")
        print("=" * 70)
        a1_hts: List[Dict[str, Any]] = []
        a1_tf: List[Dict[str, Any]] = []
        bf = make_batch_fn(max_length, num_classes, full_tasks)
        for seed in seeds:
            print(f"\n  Seed {seed}...")
            r_hts = train_one("HtS-B12", lambda: HtSB12Classifier(cfg), bf, bf, tc, seed, args.eval_batches)
            a1_hts.append(r_hts)
            print_run("HtS-B12", r_hts)
            r_tf = train_one("Transformer", lambda: TransformerClassifier(cfg), bf, bf, tc, seed, args.eval_batches)
            a1_tf.append(r_tf)
            print_run("Transformer", r_tf)
        s_hts = agg(a1_hts)
        s_tf = agg(a1_tf)
        all_summary["A1"] = {"hts": s_hts, "tf": s_tf}
        all_runs.extend(a1_hts + a1_tf)
        print(f"\n A1 SUMMARY ({len(seeds)} seeds):")
        print(f" HtS-B12:    val={s_hts['val_mean']:.1f}+/-{s_hts['val_std']:.1f}% test={s_hts['test_mean']:.1f}+/-{s_hts['test_std']:.1f}% spikes={s_hts['avg_spikes']}")
        print(f" Transformer: val={s_tf['val_mean']:.1f}+/-{s_tf['val_std']:.1f}% test={s_tf['test_mean']:.1f}+/-{s_tf['test_std']:.1f}% spikes={s_tf['avg_spikes']}")
        print(f" Delta test: {s_hts['test_mean'] - s_tf['test_mean']:+.1f}pp")

    print("\n" + "=" * 70)
    print("  A2: HELD-OUT GENERALIZATION")
    print("=" * 70)
    a2_seeds = seeds[: min(3, len(seeds))]

    print("\n A2a: Train len 1-64, test mixed len 1-200")
    a2a_hts: List[Dict[str, Any]] = []
    a2a_tf: List[Dict[str, Any]] = []
    for seed in a2_seeds:
        short_len = min(64, max_length)
        train_fn = make_batch_fn(short_len, num_classes, full_tasks)
        eval_fn = make_batch_fn(max_length, num_classes, full_tasks)
        r_hts = train_one("HtS-B12", lambda: HtSB12Classifier(cfg), train_fn, eval_fn, tc, seed, args.eval_batches)
        r_tf = train_one("Transformer", lambda: TransformerClassifier(cfg), train_fn, eval_fn, tc, seed, args.eval_batches)
        a2a_hts.append(r_hts)
        a2a_tf.append(r_tf)
        print(f" seed={seed}: HtS={r_hts['test']*100:.1f}% TF={r_tf['test']*100:.1f}%")
    s2a_h = agg(a2a_hts)
    s2a_t = agg(a2a_tf)
    all_summary["A2a"] = {"hts": s2a_h, "tf": s2a_t}
    all_runs.extend(a2a_hts + a2a_tf)
    print(f" HtS test(mixed-long): {s2a_h['test_mean']:.1f}+/-{s2a_h['test_std']:.1f}% TF: {s2a_t['test_mean']:.1f}+/-{s2a_t['test_std']:.1f}%")

    print("\n A2b: Train count_a/b, test count_digit/vowel_like")
    a2b_hts: List[Dict[str, Any]] = []
    a2b_tf: List[Dict[str, Any]] = []
    for seed in a2_seeds:
        task_len = min(128, max_length)
        train_fn = make_batch_fn(task_len, num_classes, ("count_a", "count_b"))
        eval_fn = make_batch_fn(task_len, num_classes, ("count_digit", "count_vowel_like"))
        r_hts = train_one("HtS-B12", lambda: HtSB12Classifier(cfg), train_fn, eval_fn, tc, seed, args.eval_batches)
        r_tf = train_one("Transformer", lambda: TransformerClassifier(cfg), train_fn, eval_fn, tc, seed, args.eval_batches)
        a2b_hts.append(r_hts)
        a2b_tf.append(r_tf)
        print(f" seed={seed}: HtS={r_hts['test']*100:.1f}% TF={r_tf['test']*100:.1f}%")
    s2b_h = agg(a2b_hts)
    s2b_t = agg(a2b_tf)
    all_summary["A2b"] = {"hts": s2b_h, "tf": s2b_t}
    all_runs.extend(a2b_hts + a2b_tf)
    print(f" HtS test(new tasks): {s2b_h['test_mean']:.1f}+/-{s2b_h['test_std']:.1f}% TF: {s2b_t['test_mean']:.1f}+/-{s2b_t['test_std']:.1f}%")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(" VALIDATION-A FINAL RESULTS")
    print("=" * 70)
    print(f" Total time: {elapsed/60:.1f} min")
    print(f" Device: {device}")
    print(f" HtS-B12 params: {hts_params:,} | Transformer params: {tf_params:,}")
    for k, v in all_summary.items():
        print(f"\n {k}:")
        print(f"   HtS-B12:    {v['hts']['test_mean']:.1f}+/-{v['hts']['test_std']:.1f}%")
        print(f"   Transformer: {v['tf']['test_mean']:.1f}+/-{v['tf']['test_std']:.1f}%")
        print(f"   Delta:       {v['hts']['test_mean'] - v['tf']['test_mean']:+.1f}pp")

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(all_summary, f, indent=2)
    with open(RESULTS_DIR / "all_runs.csv", "w", newline="") as f:
        fieldnames = ["model", "seed", "params", "best_val", "best_step", "test", "spikes", "max_spike", "final_loss"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_runs:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\n Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
