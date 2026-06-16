"""Dual-GPU launcher for Benchmark-D OOD-Fix on Kaggle T4x2.

Runs the original HtS-B12 dense class head and the new HtS-DigitOOD head in
separate processes/GPU devices, then merges CSV outputs.

Default mapping:
  - GPU 0: HtS-B12-ClassHead baseline
  - GPU 1: HtS-DigitOOD fix
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
from typing import Any, Dict, List

import torch

BASE = Path(__file__).resolve().parent
SINGLE = BASE / "kaggle_publication_d_oodfix.py"
DEFAULT_OUT = Path("publication_d_oodfix_results_dualgpu")
TMP_CLASS = Path("publication_d_oodfix_class_gpu0")
TMP_DIGIT = Path("publication_d_oodfix_digit_gpu1")


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists(): return []
    with path.open(newline="") as f: return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8"); return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in keys})


def launch(label: str, gpu: int, model_filter: str, out_dir: Path, args: argparse.Namespace) -> subprocess.Popen:
    cmd=[sys.executable, str(SINGLE),
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
        "--output-dir", str(out_dir)]
    if args.quick: cmd.append("--quick")
    env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu); env.setdefault("PYTHONUNBUFFERED","1")
    print("\n"+"="*92, flush=True)
    print(f"Launching {label} on physical GPU {gpu}: CUDA_VISIBLE_DEVICES={gpu}", flush=True)
    print(" ".join(cmd), flush=True)
    print("="*92+"\n", flush=True)
    return subprocess.Popen(cmd, env=env)


def is_missing(x: Any) -> bool:
    if x is None: return True
    s=str(x).strip()
    return s=="" or s.lower() in {"nan","none","null"}


def ff(x: Any, n: int=2) -> str:
    if is_missing(x): return "—"
    try: return f"{float(x):.{n}f}"
    except Exception: return str(x)


def make_result_card(summary: List[Dict[str,str]], config: Dict[str,Any], path: Path) -> None:
    lines=["# HtS-B12 Benchmark-D Dual-GPU OOD-Fix Result Card", "", "## Dual-GPU execution", "- GPU 0: `HtS-B12-ClassHead` baseline", "- GPU 1: `HtS-DigitOOD` compositional numeric-head fix", "", "## Protocol"]
    for k in ["seeds","steps","batch_size","eval_batches","protocols","train_max_length","max_eval_length","num_classes","holdout_mod"]:
        lines.append(f"- {k}: `{config.get(k)}`")
    protos=[]
    for r in summary:
        p=r.get("protocol","")
        if p and p not in protos: protos.append(p)
    lines.append("\n## Summary")
    for p in protos:
        rows=[r for r in summary if r.get("protocol")==p]
        suites=[]
        for r in rows:
            for k,v in r.items():
                if k.endswith("_acc_mean") and not is_missing(v):
                    pref=k[:-9]
                    if pref not in suites: suites.append(pref)
        lines.append(f"\n### {p}")
        header=["Model","Params","Val"]+[f"{s} Acc" for s in suites]+["Avg spikes"]
        lines.append("| "+" | ".join(header)+" |")
        lines.append("|"+"|".join(["---"]*len(header))+"|")
        for r in rows:
            try: params=f"{int(float(r.get('params','0'))):,}"
            except Exception: params=str(r.get('params',''))
            row=[r.get("model",""), params, f"{ff(r.get('val_mean'))} ± {ff(r.get('val_std'))}%"]
            for s in suites:
                row.append(f"{ff(r.get(s+'_acc_mean'))} ± {ff(r.get(s+'_acc_std'))}%" if not is_missing(r.get(s+'_acc_mean')) else "—")
            row.append(ff(r.get("avg_spikes"),3))
            lines.append("| "+" | ".join(row)+" |")
    lines += ["", "## Key test", "- Main target: `HtS-DigitOOD` should improve `d1_c3_extrapolation_fix/extrap_all`, `extrap_length_only`, and/or `extrap_count_only` over `HtS-B12-ClassHead` without training on 129..200.", "- If `extrap_length_only` improves but `extrap_count_only` remains low, the output-head bottleneck is partly fixed and the next bottleneck is count accumulation."]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--seeds", type=str, default="42,123,777")
    ap.add_argument("--protocols", type=str, default="d1,d2")
    ap.add_argument("--train-max-length", type=int, default=128)
    ap.add_argument("--max-eval-length", type=int, default=200)
    ap.add_argument("--num-classes", type=int, default=256)
    ap.add_argument("--holdout-mod", type=int, default=7)
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--keep-temp", action="store_true")
    args=ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:
        raise SystemExit(f"Dual-GPU mode requires >=2 CUDA GPUs; found {torch.cuda.device_count() if torch.cuda.is_available() else 0}.")
    out=Path(args.output_dir)
    for d in [out,TMP_CLASS,TMP_DIGIT]:
        if d.exists() and not args.keep_temp: shutil.rmtree(d)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Detected CUDA GPUs: {torch.cuda.device_count()}", flush=True)
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} VRAM={p.total_memory/1e9:.1f} GB", flush=True)
    t0=time.time()
    procs=[
        ("HtS-B12-ClassHead", launch("HtS-B12-ClassHead",0,"class",TMP_CLASS,args)),
        ("HtS-DigitOOD", launch("HtS-DigitOOD",1,"digit",TMP_DIGIT,args)),
    ]
    failed=[]
    for label,p in procs:
        rc=p.wait()
        if rc!=0: failed.append(f"{label} exited with code {rc}")
    if failed: raise SystemExit("\n".join(failed))
    runs=[]; summ=[]; curves=[]
    for d in [TMP_CLASS,TMP_DIGIT]:
        runs.extend(read_csv(d/"all_runs.csv")); summ.extend(read_csv(d/"summary.csv")); curves.extend(read_csv(d/"training_curves.csv"))
    write_csv(out/"all_runs.csv", runs); write_csv(out/"summary.csv", summ); write_csv(out/"training_curves.csv", curves)
    config={"dual_gpu":True,"classhead_physical_gpu":0,"digitood_physical_gpu":1,"seeds":[int(s.strip()) for s in args.seeds.split(',') if s.strip()],"steps":args.steps,"batch_size":args.batch_size,"eval_batches":args.eval_batches,"protocols":args.protocols,"train_max_length":args.train_max_length,"max_eval_length":args.max_eval_length,"num_classes":args.num_classes,"holdout_mod":args.holdout_mod,"models":["HtS-B12-ClassHead","HtS-DigitOOD"]}
    (out/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    make_result_card(summ, config, out/"result_card.md")
    print("\n"+"="*92); print("DUAL-GPU BENCHMARK-D MERGED SUMMARY"); print("="*92)
    for r in summ:
        print(f"[{r.get('protocol')}] {r.get('model')} params={r.get('params')} val={r.get('val_mean')}±{r.get('val_std')} avg_spikes={r.get('avg_spikes')}")
        for k,v in r.items():
            if k.endswith('_acc_mean'):
                s=k[:-9]
                print(f"  {s:<24} {v}% ± {r.get(s+'_acc_std','')}%")
    print(f"\nSaved merged results to: {out.resolve()}")
    print(f"Elapsed wall-clock: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__": main()
