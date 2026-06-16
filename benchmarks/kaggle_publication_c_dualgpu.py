"""Dual-GPU launcher for Publication Benchmark-C on Kaggle T4x2.

Runs HtS-B12 on physical GPU 0 and Transformer-ParamMatched on physical GPU 1
in two independent Python processes, then merges result CSVs into one result
folder.  This is intentionally process-based rather than DataParallel: HtS and
Transformer are independent baselines, so running one model per GPU is simpler,
faster, and avoids cross-model memory interference.

Example Kaggle commands
-----------------------
Smoke:
    !python /kaggle/working/HtS/benchmarks/kaggle_publication_c_dualgpu.py --quick

C1 on two GPUs:
    !python /kaggle/working/HtS/benchmarks/kaggle_publication_c_dualgpu.py \
        --protocols c1 --steps 5000 --seeds 42,123,777

Full diagnostic run:
    !python /kaggle/working/HtS/benchmarks/kaggle_publication_c_dualgpu.py \
        --protocols c1,c2,c3 --steps 5000 --seeds 42,123,777

Outputs:
    publication_c_results_dualgpu/summary.csv
    publication_c_results_dualgpu/all_runs.csv
    publication_c_results_dualgpu/training_curves.csv
    publication_c_results_dualgpu/result_card.md
    publication_c_results_dualgpu/config.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

BASE = Path(__file__).resolve().parent
SINGLE_SCRIPT = BASE / "kaggle_publication_c.py"
DEFAULT_OUT = Path("publication_c_results_dualgpu")
TMP_HTS = Path("publication_c_results_dualgpu_hts_gpu0")
TMP_TFM = Path("publication_c_results_dualgpu_tfm_gpu1")
TMP_NOSOFT = Path("publication_c_results_dualgpu_nosoft_gpu0")


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def launch(label: str, physical_gpu: int, model_filter: str, out_dir: Path, args: argparse.Namespace) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(SINGLE_SCRIPT),
        "--steps", str(args.steps),
        "--batch-size", str(args.batch_size),
        "--eval-batches", str(args.eval_batches),
        "--seeds", args.seeds,
        "--protocols", args.protocols,
        "--train-max-length", str(args.train_max_length),
        "--max-eval-length", str(args.max_eval_length),
        "--num-classes", str(args.num_classes),
        "--holdout-mod", str(args.holdout_mod),
        "--model-filter", model_filter,
        "--device", "cuda",
        "--output-dir", str(out_dir),
    ]
    if args.quick:
        cmd.append("--quick")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")
    print("\n" + "=" * 88, flush=True)
    print(f"Launching {label} on physical GPU {physical_gpu}: CUDA_VISIBLE_DEVICES={physical_gpu}", flush=True)
    print(" ".join(cmd), flush=True)
    print("=" * 88 + "\n", flush=True)
    return subprocess.Popen(cmd, env=env)


def make_result_card(summary_rows: List[Dict[str, str]], config: Dict[str, Any], out_path: Path) -> None:
    """Write a robust Markdown result card from merged CSV rows.

    Some protocol-specific metrics are intentionally blank in the merged CSV
    (e.g., C1 has full_id, C2 has heldout_buckets, C3 has extrap_length).
    This function therefore treats '', None, and NaN-like strings as missing
    rather than trying to convert them to floats.
    """
    def is_missing(x: Any) -> bool:
        if x is None:
            return True
        xs = str(x).strip()
        return xs == "" or xs.lower() in {"nan", "none", "null"}

    def fmt_float(x: Any, ndigits: int = 2, suffix: str = "") -> str:
        if is_missing(x):
            return "—"
        try:
            return f"{float(x):.{ndigits}f}{suffix}"
        except Exception:
            return str(x)

    def fmt_acc(row: Dict[str, str], prefix: str) -> str:
        mean = row.get(f"{prefix}_acc_mean", "")
        std = row.get(f"{prefix}_acc_std", "")
        if is_missing(mean):
            return "—"
        if is_missing(std):
            return f"{fmt_float(mean, 2)}%"
        return f"{fmt_float(mean, 2)} ± {fmt_float(std, 2)}%"

    def fmt_params(x: Any) -> str:
        if is_missing(x):
            return "—"
        try:
            return f"{int(float(x)):,}"
        except Exception:
            return str(x)

    lines: List[str] = []
    lines.append("# HtS-B12 Publication Benchmark-C Dual-GPU Result Card")
    lines.append("")
    lines.append("## Dual-GPU execution")
    lines.append("- HtS-B12 process: physical GPU `0` via `CUDA_VISIBLE_DEVICES=0`")
    lines.append("- Transformer-ParamMatched process: physical GPU `1` via `CUDA_VISIBLE_DEVICES=1`")
    if config.get("include_ablation"):
        lines.append("- HtS-NoSoft ablation process: physical GPU `0` after/alongside HtS path")
    lines.append("")
    lines.append("## Protocol")
    for k in ["seeds", "steps", "batch_size", "eval_batches", "protocols", "train_max_length", "max_eval_length", "num_classes", "holdout_mod"]:
        lines.append(f"- {k}: `{config.get(k)}`")
    lines.append("")

    protocols: List[str] = []
    for r in summary_rows:
        p = r.get("protocol", "")
        if p and p not in protocols:
            protocols.append(p)

    preferred = {
        "c1_seen_full_range": ["full_id", "full_length_only", "full_count_only", "full_biased_count", "full_rare_count"],
        "c2_bucket_interpolation": ["seen_buckets", "heldout_buckets", "heldout_length_only", "heldout_count_only"],
        "c3_extrapolation": ["id_1_trainmax", "extrap_length", "extrap_length_only", "extrap_count_only", "extrap_biased_count"],
    }

    lines.append("## Summary")
    for p in protocols:
        rows = [r for r in summary_rows if r.get("protocol") == p]
        lines.append(f"\n### {p}")
        if not rows:
            continue

        # Use known protocol order when available; otherwise infer non-empty metric prefixes.
        if p in preferred:
            suite_prefixes = preferred[p]
        else:
            seen = []
            for r in rows:
                for k, v in r.items():
                    if k.endswith("_acc_mean") and not is_missing(v):
                        prefix = k[:-9]
                        if prefix not in seen:
                            seen.append(prefix)
            suite_prefixes = seen

        header = ["Model", "Params", "Val"] + [f"{s} Acc" for s in suite_prefixes] + ["Avg spikes"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in rows:
            row = [
                r.get("model", ""),
                fmt_params(r.get("params", "")),
                f"{fmt_float(r.get('val_mean', ''), 2)} ± {fmt_float(r.get('val_std', ''), 2)}%" if not is_missing(r.get("val_mean", "")) else "—",
            ]
            for s in suite_prefixes:
                row.append(fmt_acc(r, s))
            row.append(fmt_float(r.get("avg_spikes", ""), 3))
            lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("- Use C1 to test full-range capacity and label availability.")
    lines.append("- Use C2 to test bucket interpolation inside the seen positional range; `heldout_buckets` is the key C2 metric.")
    lines.append("- Use C3 to test true length extrapolation beyond the training length range; `extrap_length` is the key C3 metric.")
    lines.append("- Blank protocol-specific metrics are rendered as `—` instead of crashing the merger.")
    lines.append("- Because HtS and Transformer are trained concurrently on different GPUs, wall-clock time is reduced, but numerical comparisons remain independent per model.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--seeds", type=str, default="42,123,777")
    parser.add_argument("--protocols", type=str, default="c1,c2,c3")
    parser.add_argument("--train-max-length", type=int, default=128)
    parser.add_argument("--max-eval-length", type=int, default=200)
    parser.add_argument("--num-classes", type=int, default=256)
    parser.add_argument("--holdout-mod", type=int, default=7)
    parser.add_argument("--include-ablation", action="store_true", help="Also run HtS-NoSoft on GPU 0; slower.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    if args.num_classes <= args.max_eval_length:
        raise SystemExit("--num-classes must exceed --max-eval-length to avoid label clipping.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit(
            f"Dual-GPU mode requires at least 2 visible CUDA GPUs. Found {torch.cuda.device_count() if torch.cuda.is_available() else 0}."
        )

    out_dir = Path(args.output_dir)
    for d in [out_dir, TMP_HTS, TMP_TFM, TMP_NOSOFT]:
        if d.exists() and not args.keep_temp:
            shutil.rmtree(d)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Detected CUDA GPUs: {torch.cuda.device_count()}", flush=True)
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} VRAM={props.total_memory/1e9:.1f} GB", flush=True)

    t0 = time.time()
    procs = [
        ("HtS-B12", launch("HtS-B12", 0, "hts-b12", TMP_HTS, args)),
        ("Transformer-ParamMatched", launch("Transformer-ParamMatched", 1, "transformer", TMP_TFM, args)),
    ]
    if args.include_ablation:
        # HtS-NoSoft uses the same model family, so put it on GPU 0.  It runs concurrently
        # with Transformer on GPU 1, but may share GPU0 with HtS if both are active.  For
        # strict timing, run without --include-ablation first, then ablation separately.
        procs.append(("HtS-NoSoft", launch("HtS-NoSoft", 0, "no-soft", TMP_NOSOFT, args)))

    failed: List[str] = []
    for label, p in procs:
        rc = p.wait()
        if rc != 0:
            failed.append(f"{label} exited with code {rc}")
    if failed:
        raise SystemExit("Dual-GPU run failed:\n" + "\n".join(failed))

    all_runs: List[Dict[str, str]] = []
    summary: List[Dict[str, str]] = []
    curves: List[Dict[str, str]] = []
    for d in [TMP_HTS, TMP_TFM, TMP_NOSOFT if args.include_ablation else None]:
        if d is None:
            continue
        all_runs.extend(read_csv(d / "all_runs.csv"))
        summary.extend(read_csv(d / "summary.csv"))
        curves.extend(read_csv(d / "training_curves.csv"))

    write_csv(out_dir / "all_runs.csv", all_runs)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "training_curves.csv", curves)
    config = {
        "dual_gpu": True,
        "hts_physical_gpu": 0,
        "transformer_physical_gpu": 1,
        "include_ablation": bool(args.include_ablation),
        "seeds": [int(s.strip()) for s in args.seeds.split(",") if s.strip()],
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_batches": args.eval_batches,
        "protocols": args.protocols,
        "train_max_length": args.train_max_length,
        "max_eval_length": args.max_eval_length,
        "num_classes": args.num_classes,
        "holdout_mod": args.holdout_mod,
        "models": ["HtS-B12", "Transformer-ParamMatched"] + (["HtS-NoSoft"] if args.include_ablation else []),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    make_result_card(summary, config, out_dir / "result_card.md")

    print("\n" + "=" * 88)
    print("DUAL-GPU BENCHMARK-C MERGED SUMMARY")
    print("=" * 88)
    for r in summary:
        print(f"[{r.get('protocol')}] {r.get('model')} params={r.get('params')} val={r.get('val_mean')}±{r.get('val_std')} avg_spikes={r.get('avg_spikes')}")
        for k, v in r.items():
            if k.endswith("_acc_mean"):
                s = k[:-9]
                print(f"  {s:<24} {v}% ± {r.get(s + '_acc_std', '')}%")
    print(f"\nSaved merged results to: {out_dir.resolve()}")
    print(f"Elapsed wall-clock: {(time.time() - t0)/60:.1f} min")
    print("Note: HtS and Transformer were trained concurrently on separate physical GPUs.")


if __name__ == "__main__":
    main()
