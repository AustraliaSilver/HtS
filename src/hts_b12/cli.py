"""Command line interface for HtS-B12."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import HtSB12Config, TrainConfig
from .data import make_string_count_batch
from .device import detect_device
from .diagnostics import count_parameters
from .groups import ModelGroupConfig, build_hts_config_from_group
from .losses import HtSB12Objective
from .modeling import HtSB12Classifier, TransformerClassifier
from .training import train_classifier


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train or inspect HtS-B12.")
    p.add_argument("--model", choices=["hts-b12", "transformer"], default="hts-b12")
    p.add_argument("--group-config", default=None, help="Optional JSON/YAML ModelGroupConfig. If omitted, uses built-in string/count settings.")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda/gpu, mps, tpu")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=32)
    p.add_argument("--num-classes", type=int, default=64)
    p.add_argument("--vocab-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dim-ff", type=int, default=256)
    p.add_argument("--rank-main", type=int, default=8)
    p.add_argument("--rank-corr", type=int, default=4)
    p.add_argument("--task-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="runs/hts_b12_demo")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    info = detect_device(args.device)

    if args.group_config:
        group = ModelGroupConfig.load(args.group_config)
        cfg = build_hts_config_from_group(
            group,
            d_model=args.d_model,
            n_heads=args.heads,
            num_layers=args.layers,
            dim_ff=args.dim_ff,
            rank_main=args.rank_main,
            rank_corr=args.rank_corr,
            task_dim=args.task_dim,
        )
        max_length = group.max_length
        num_classes = group.num_classes
        num_tasks = group.num_tasks
        print(f"Loaded group: {group.name} | tasks={num_tasks} | classes={num_classes}")
    else:
        cfg = HtSB12Config(
            vocab_size=args.vocab_size,
            num_tasks=8,
            num_classes=args.num_classes,
            max_length=args.max_length,
            d_model=args.d_model,
            n_heads=args.heads,
            num_layers=args.layers,
            dim_ff=args.dim_ff,
            rank_main=args.rank_main,
            rank_corr=args.rank_corr,
            task_dim=args.task_dim,
        )
        max_length = args.max_length
        num_classes = args.num_classes

    model = HtSB12Classifier(cfg) if args.model == "hts-b12" else TransformerClassifier(cfg)
    print(f"Backend: {info.backend} | Device: {info.device} | Name: {info.name}")
    print(f"Model: {args.model} | Params: {count_parameters(model):,}")

    def batch_fn(batch_size, device, seed):
        # Built-in quick CLI batch. For fully custom datasets, use Python API
        # with ModelGroupRegistry and a custom batch factory.
        return make_string_count_batch(
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            num_classes=num_classes,
            seed=seed,
        )

    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    objective = HtSB12Objective(
        margin=cfg.margin,
        margin_weight=cfg.margin_weight if args.model == "hts-b12" else 0.0,
        ratio_reg=cfg.ratio_reg if args.model == "hts-b12" else 0.0,
        warmup_steps=args.warmup_steps,
    )
    log = train_classifier(model, batch_fn, train_cfg, objective)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
        if log.rows:
            writer = csv.DictWriter(f, fieldnames=sorted(log.rows[0].keys()))
            writer.writeheader()
            writer.writerows(log.rows)
    print(f"Best accuracy: {100 * log.best_acc:.2f}% at step {log.best_step}")
    print(f"Saved log: {out / 'train_log.csv'}")


if __name__ == "__main__":
    main()
