from __future__ import annotations
import argparse
from pathlib import Path
from .config import HtSConfig, TransformerConfig, TrainConfig
from .training import train_synthetic


def main(argv=None):
    p = argparse.ArgumentParser(description="Train HtS or Transformer on synthetic task-conditioned benchmark.")
    p.add_argument("--model", default="hts", choices=["hts", "transformer"], help="Model kind.")
    p.add_argument("--benchmark", default="synthetic", choices=["synthetic", "multi_step", "compositional", "string_length"],
                   help="Benchmark: synthetic (original 24 tasks), multi_step (chain reasoning), compositional (task arithmetic), string_length (string length prediction).")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, tpu")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="runs/hts_synthetic")
    p.add_argument("--d-model", type=int, default=40)
    p.add_argument("--dim-ff", type=int, default=64)
    p.add_argument("--rank-main", type=int, default=5)
    p.add_argument("--rank-corr", type=int, default=2)
    p.add_argument("--layers", type=int, default=1)
    args = p.parse_args(argv)

    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        benchmark=args.benchmark,
    )
    hts_cfg = HtSConfig(d_model=args.d_model, dim_ff=args.dim_ff, rank_main=args.rank_main, rank_corr=args.rank_corr, n_layers=args.layers)
    tf_cfg = TransformerConfig(d_model=args.d_model, dim_ff=args.dim_ff, n_layers=args.layers)
    result = train_synthetic(args.model, train_cfg, hts_cfg, tf_cfg, args.out_dir)
    print("Finished.")
    print(result["meta"])
    print("Metrics:", result["metrics_path"])


if __name__ == "__main__":
    main()
