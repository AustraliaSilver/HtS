#!/usr/bin/env python3
"""Quick String Length / Count benchmark for HtS-B12.

Example:
    python benchmarks/string_length_count.py --model hts-b12 --steps 1000 --device auto
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hts_b12 import HtSB12Classifier, HtSB12Config, HtSB12Objective, TransformerClassifier, TrainConfig, count_parameters
from hts_b12.data import make_string_count_batch
from hts_b12.training import train_classifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["hts-b12", "transformer"], default="hts-b12")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/string_length_count")
    args = ap.parse_args()

    cfg = HtSB12Config(
        vocab_size=128,
        num_tasks=8,
        num_classes=max(128, args.max_length + 1),
        max_length=args.max_length,
        d_model=128,
        n_heads=4,
        num_layers=2,
        dim_ff=256,
        rank_main=8,
        rank_corr=4,
    )
    model = HtSB12Classifier(cfg) if args.model == "hts-b12" else TransformerClassifier(cfg)
    print(f"model={args.model} params={count_parameters(model):,}")

    def batch_fn(batch_size, device, seed):
        return make_string_count_batch(batch_size, args.max_length, device, num_classes=cfg.num_classes, seed=seed)

    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=1e-3,
        warmup_steps=max(10, args.steps // 20),
        eval_every=max(1, args.steps // 10),
        device=args.device,
        seed=args.seed,
        output_dir=args.out,
    )
    objective = HtSB12Objective(
        margin=cfg.margin,
        margin_weight=cfg.margin_weight if args.model == "hts-b12" else 0.0,
        ratio_reg=cfg.ratio_reg if args.model == "hts-b12" else 0.0,
        warmup_steps=train_cfg.warmup_steps,
    )
    log = train_classifier(model, batch_fn, train_cfg, objective)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{args.model}_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(log.rows[0].keys()))
        writer.writeheader()
        writer.writerows(log.rows)
    print(f"best_acc={100*log.best_acc:.2f}% best_step={log.best_step}")


if __name__ == "__main__":
    main()
