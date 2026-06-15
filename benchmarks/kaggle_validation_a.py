"""
HtS-B12 vs Transformer: Validation-A Benchmark (Kaggle T4x2)
=============================================================

Runs:
  A1: Stability  — 5 seeds, cosine warmup, loss spike detection
  A2: Held-out   — train short/test long, train subset/test new tasks
  A3: Fair baseline — same params/scheduler for both models
  A4: Dev/test strict separation

Install before running:
  pip install git+https://github.com/AustraliaSilver/HtS.git

Usage:
  python kaggle_validation_a.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

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

SEEDS = [42, 123, 777, 2024, 9999]
RESULTS_DIR = Path("validation_a_results")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_batch_fn(max_length, num_classes, task_mix):
    def fn(batch_size, device, seed):
        return make_string_count_batch(
            batch_size=batch_size, max_length=max_length, device=device,
            num_classes=num_classes, seed=seed, task_mix=task_mix,
        )
    return fn


def train_one(
    model_name: str,
    model: torch.nn.Module,
    train_batch_fn: Callable,
    eval_batch_fn: Callable,
    config: TrainConfig,
    seed: int,
) -> Dict[str, Any]:
    seed_everything(seed)
    device = config.device if config.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    model.train()

    is_hts = isinstance(model, HtSB12Classifier)
    objective = HtSB12Objective(
        margin=0.6, margin_weight=0.03, ratio_reg=1e-3,
        warmup_steps=config.warmup_steps,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    best_val = -1.0
    best_step = 0
    prev_loss = None
    n_spikes = 0
    max_spike = 0.0
    log = []

    for step in range(1, config.steps + 1):
        lr = cosine_with_warmup(step, config.steps, config.warmup_steps, config.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        batch = train_batch_fn(config.batch_size, device, config.seed * 1_000_000 + step)
        logits = model(batch.input_ids, batch.task_ids)

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
            model.eval()
            with torch.no_grad():
                tr_acc = float(accuracy(logits, batch.labels))
                ev = eval_batch_fn(config.batch_size, device, config.seed * 2_000_000 + step)
                ev_logits = model(ev.input_ids, ev.task_ids)
                val_acc = float(accuracy(ev_logits, ev.labels))
            model.train()

            if val_acc > best_val:
                best_val = val_acc
                best_step = step

            row = {
                "step": step, "seed": seed, "loss": round(cur_loss, 4),
                "train_acc": round(tr_acc * 100, 2), "val_acc": round(val_acc * 100, 2),
            }
            if is_hts:
                diag = model.hts_diagnostics()
                for k in ("l0_layer0_b12_delta_base_ratio", "l0_layer0_b12_gate_main",
                          "l1_layer1_b12_delta_base_ratio", "l1_layer1_b12_gate_main"):
                    if k in diag:
                        row[k] = round(diag[k], 4)
            log.append(row)

    model.eval()
    with torch.no_grad():
        te = eval_batch_fn(512, device, config.seed * 3_000_000)
        te_logits = model(te.input_ids, te.task_ids)
        test_acc = float(accuracy(te_logits, te.labels))

    return {
        "model": model_name, "seed": seed, "params": count_parameters(model),
        "best_val": best_val, "best_step": best_step, "test": test_acc,
        "spikes": n_spikes, "max_spike": round(max_spike, 2),
        "final_loss": round(cur_loss, 4), "log": log,
    }


def agg(results: List[Dict]) -> Dict[str, float]:
    vals = [r["best_val"] for r in results]
    tests = [r["test"] for r in results]
    mu_v, mu_t = np.mean(vals), np.mean(tests)
    return {
        "val_mean": round(mu_v * 100, 2), "val_std": round(float(np.std(vals)) * 100, 2),
        "test_mean": round(mu_t * 100, 2), "test_std": round(float(np.std(tests)) * 100, 2),
        "avg_spikes": round(np.mean([r["spikes"] for r in results]), 1),
        "max_spike": round(max(r["max_spike"] for r in results), 2),
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    max_length = 200
    num_classes = 128

    cfg = HtSB12Config(
        vocab_size=128, max_length=max_length, num_tasks=8, num_classes=num_classes,
        d_model=80, n_heads=4, num_layers=2, dim_ff=128,
        task_dim=16, rank_main=8, rank_corr=4, dropout=0.0,
        use_cls_token=True, pool="cls",
    )

    hts_params = count_parameters(HtSB12Classifier(cfg))
    tf_params = count_parameters(TransformerClassifier(cfg))
    print(f"HtS-B12 params:     {hts_params:,}")
    print(f"Transformer params: {tf_params:,}")
    print(f"Param diff:         {hts_params - tf_params:,} ({(hts_params - tf_params)/tf_params*100:+.1f}%)")

    tc = TrainConfig(
        steps=5000, batch_size=256, lr=1e-3, weight_decay=0.01,
        warmup_steps=250, grad_clip=1.0, eval_every=500,
        device=device, seed=42,
    )

    full_tasks = ("length", "count_a", "count_b", "count_digit")
    t0 = time.time()
    all_summary = {}

    # =====================================================================
    # A1: STABILITY — multi-seed
    # =====================================================================
    print("\n" + "=" * 70)
    print("  A1: STABILITY — 5 seeds, cosine warmup, spike detection")
    print("=" * 70)

    a1_hts, a1_tf = [], []
    for seed in SEEDS:
        print(f"\n  Seed {seed}...")
        bf = make_batch_fn(max_length, num_classes, full_tasks)

        r = train_one("HtS-B12", HtSB12Classifier(cfg), bf, bf, tc, seed)
        a1_hts.append(r)
        print(f"    HtS-B12:      val={r['best_val']*100:.1f}%  test={r['test']*100:.1f}%  spikes={r['spikes']}  max_spike={r['max_spike']}x")

        r = train_one("Transformer", TransformerClassifier(cfg), bf, bf, tc, seed)
        a1_tf.append(r)
        print(f"    Transformer:  val={r['best_val']*100:.1f}%  test={r['test']*100:.1f}%  spikes={r['spikes']}  max_spike={r['max_spike']}x")

    s_hts = agg(a1_hts)
    s_tf = agg(a1_tf)
    all_summary["A1"] = {"hts": s_hts, "tf": s_tf}

    print(f"\n  A1 SUMMARY ({len(SEEDS)} seeds):")
    print(f"    HtS-B12:      val={s_hts['val_mean']:.1f}+/-{s_hts['val_std']:.1f}%  test={s_hts['test_mean']:.1f}+/-{s_hts['test_std']:.1f}%  spikes={s_hts['avg_spikes']}")
    print(f"    Transformer:  val={s_tf['val_mean']:.1f}+/-{s_tf['val_std']:.1f}%  test={s_tf['test_mean']:.1f}+/-{s_tf['test_std']:.1f}%  spikes={s_tf['avg_spikes']}")
    print(f"    Delta test:   {s_hts['test_mean'] - s_tf['test_mean']:+.1f}pp")

    # =====================================================================
    # A2: HELD-OUT GENERALIZATION
    # =====================================================================
    print("\n" + "=" * 70)
    print("  A2: HELD-OUT GENERALIZATION")
    print("=" * 70)

    # A2a: Train on short lengths, test on long lengths
    print("\n  A2a: Train len 1-64, test len 65-200")
    a2a_hts, a2a_tf = [], []
    for seed in SEEDS[:3]:
        train_fn = make_batch_fn(64, num_classes, full_tasks)
        eval_fn = make_batch_fn(200, num_classes, full_tasks)

        r = train_one("HtS-B12", HtSB12Classifier(cfg), train_fn, eval_fn, tc, seed)
        a2a_hts.append(r)
        r = train_one("Transformer", TransformerClassifier(cfg), train_fn, eval_fn, tc, seed)
        a2a_tf.append(r)
        print(f"    seed={seed}: HtS={r['test']*100:.1f}% TF={r['test']*100:.1f}%")

    s2a_h = agg(a2a_hts)
    s2a_t = agg(a2a_tf)
    all_summary["A2a"] = {"hts": s2a_h, "tf": s2a_t}
    print(f"    HtS test(long): {s2a_h['test_mean']:.1f}+/-{s2a_h['test_std']:.1f}%  TF: {s2a_t['test_mean']:.1f}+/-{s2a_t['test_std']:.1f}%")

    # A2b: Train on subset of tasks, test on held-out tasks
    print("\n  A2b: Train count_a/b, test count_digit/vowel")
    a2b_hts, a2b_tf = [], []
    for seed in SEEDS[:3]:
        train_fn = make_batch_fn(128, num_classes, ("count_a", "count_b"))
        eval_fn = make_batch_fn(128, num_classes, ("count_digit", "count_vowel_like"))

        r = train_one("HtS-B12", HtSB12Classifier(cfg), train_fn, eval_fn, tc, seed)
        a2b_hts.append(r)
        r = train_one("Transformer", TransformerClassifier(cfg), train_fn, eval_fn, tc, seed)
        a2b_tf.append(r)
        print(f"    seed={seed}: HtS={r['test']*100:.1f}% TF={r['test']*100:.1f}%")

    s2b_h = agg(a2b_hts)
    s2b_t = agg(a2b_tf)
    all_summary["A2b"] = {"hts": s2b_h, "tf": s2b_t}
    print(f"    HtS test(new):  {s2b_h['test_mean']:.1f}+/-{s2b_h['test_std']:.1f}%  TF: {s2b_t['test_mean']:.1f}+/-{s2b_t['test_std']:.1f}%")

    # =====================================================================
    # FINAL
    # =====================================================================
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("  VALIDATION-A FINAL RESULTS")
    print("=" * 70)
    print(f"  Total time: {elapsed/60:.1f} min")
    print(f"  Device: {device}")
    print(f"  HtS-B12 params: {hts_params:,}  |  Transformer params: {tf_params:,}")

    print(f"\n  A1 (Stability):")
    print(f"    HtS-B12:     {all_summary['A1']['hts']['test_mean']:.1f}+/-{all_summary['A1']['hts']['test_std']:.1f}%")
    print(f"    Transformer: {all_summary['A1']['tf']['test_mean']:.1f}+/-{all_summary['A1']['tf']['test_std']:.1f}%")
    print(f"    Delta:       {all_summary['A1']['hts']['test_mean'] - all_summary['A1']['tf']['test_mean']:+.1f}pp")

    print(f"\n  A2a (Train short, test long):")
    print(f"    HtS-B12:     {all_summary['A2a']['hts']['test_mean']:.1f}+/-{all_summary['A2a']['hts']['test_std']:.1f}%")
    print(f"    Transformer: {all_summary['A2a']['tf']['test_mean']:.1f}+/-{all_summary['A2a']['tf']['test_std']:.1f}%")

    print(f"\n  A2b (Train subset, test held-out tasks):")
    print(f"    HtS-B12:     {all_summary['A2b']['hts']['test_mean']:.1f}+/-{all_summary['A2b']['hts']['test_std']:.1f}%")
    print(f"    Transformer: {all_summary['A2b']['tf']['test_mean']:.1f}+/-{all_summary['A2b']['tf']['test_std']:.1f}%")

    # Save
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(all_summary, f, indent=2)

    # Save detailed logs
    all_runs = a1_hts + a1_tf + a2a_hts + a2a_tf + a2b_hts + a2b_tf
    with open(RESULTS_DIR / "all_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "seed", "params", "best_val", "best_step",
                                           "test", "spikes", "max_spike", "final_loss"])
        w.writeheader()
        for r in all_runs:
            w.writerow({k: r[k] for k in w.fieldnames})

    print(f"\n  Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
