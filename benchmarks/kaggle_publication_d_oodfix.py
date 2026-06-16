"""Publication Benchmark-D: OOD target-parameterization fix for HtS-B12.

Benchmark-C showed:
  - HtS-B12 is strong inside the observed length support.
  - It fails on held-out length buckets and strict extrapolation.

A core reason is output parameterization.  A 256-way classifier cannot learn
unseen labels compositionally: when train labels stop at 128, target classes
129..200 never receive positive supervision.  Benchmark-D tests a minimal fix:
replace the dense class head with a compositional digit head (hundreds/tens/ones)
while keeping the HtS-B12 encoder and generated-computation blocks unchanged.

This does not leak held-out labels or train on extrapolation data.  It changes
only how numeric labels are represented at the output.
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from hts_b12 import HtSB12Classifier, HtSB12Config, HtSB12DigitClassifier, TrainConfig, accuracy, count_parameters
from hts_b12.losses import HtSB12Objective
from hts_b12.training import cosine_with_warmup

DEFAULT_SEEDS = [42, 123, 777]
RESULTS_DIR = Path("publication_d_oodfix_results")

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
TASKS = {"length": 0, "count_a": 1, "count_b": 2, "count_digit": 3}


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
        values = torch.tensor([1, 1, 1, 2, 2, 5, 6, 7, 8, 9, 10, 3, 4], dtype=torch.long)
        idx = torch.randint(0, len(values), (batch_size, max_len), generator=gen, dtype=torch.long)
        ids = values[idx]
    elif token_mode == "rare_target":
        values = torch.tensor([1, 2, 5, 6, 7, 8, 3, 3, 4, 4, 9, 9, 10, 10], dtype=torch.long)
        idx = torch.randint(0, len(values), (batch_size, max_len), generator=gen, dtype=torch.long)
        ids = values[idx]
    else:
        raise ValueError(token_mode)

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
        raise RuntimeError("label out of class range")

    return Batch(ids.to(dev), task_ids.to(dev), labels.to(dev), mask.to(dtype=torch.long).to(dev))


def batch_fn(allowed_lengths: Sequence[int], task_mix: Sequence[str], token_mode: str, num_classes: int):
    def fn(batch_size: int, device: torch.device | str, seed: int) -> Batch:
        return make_batch(batch_size, allowed_lengths, device, seed, task_mix, token_mode, num_classes)
    return fn


def build_hts_config(max_model_length: int, num_classes: int, tiny: bool = False) -> HtSB12Config:
    """Build the benchmark HtS config.

    The publication run uses the full config.  ``tiny=True`` is used only by
    CPU smoke tests so the package can be validated quickly on machines without
    a GPU.  It exercises the same code paths but does not produce scientific
    benchmark numbers.
    """
    if tiny:
        return HtSB12Config(
            vocab_size=128,
            max_length=max_model_length,
            num_tasks=8,
            num_classes=num_classes,
            d_model=16,
            n_heads=2,
            num_layers=1,
            dim_ff=32,
            task_dim=8,
            rank_main=2,
            rank_corr=1,
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


def forward_model(model: torch.nn.Module, batch: Batch):
    return model(batch.input_ids, batch.task_ids, batch.attention_mask)


def is_digit_model(model: torch.nn.Module) -> bool:
    return hasattr(model, "digit_loss") and hasattr(model, "predict_digits")


def compute_loss(model: torch.nn.Module, outputs, labels: torch.Tensor, step: int, objective: HtSB12Objective) -> torch.Tensor:
    if is_digit_model(model):
        # Digit loss + HtS regularizers.  We still include safety regularizers,
        # but not the dense-class margin loss because outputs are factored.
        loss = model.digit_loss(outputs, labels)  # type: ignore[attr-defined]
        warm = min(1.0, step / max(1, objective.warmup_steps)) if objective.warmup_steps else 1.0
        if hasattr(model, "hts_regularizers"):
            budget, binary, ratio, offset = model.hts_regularizers()
            loss = loss + objective.ratio_reg * warm * ratio
        return loss
    else:
        return objective(model, outputs, labels, step=step).loss


def eval_accuracy_and_loss(model: torch.nn.Module, fn, batch_size: int, device: torch.device | str, seed_base: int, batches: int) -> Tuple[float, float]:
    model.eval()
    accs: List[float] = []
    losses: List[float] = []
    with torch.no_grad():
        for i in range(batches):
            b = fn(batch_size, device, seed_base + i)
            out = forward_model(model, b)
            if is_digit_model(model):
                accs.append(float(model.digit_accuracy(out, b.labels)))  # type: ignore[attr-defined]
                losses.append(float(model.digit_loss(out, b.labels).detach().cpu()))  # type: ignore[attr-defined]
            else:
                accs.append(float(accuracy(out, b.labels)))
                losses.append(float(F.cross_entropy(out, b.labels).detach().cpu()))
    return float(np.mean(accs)), float(np.mean(losses))


def train_one(protocol: str, model_name: str, factory, train_fn, val_fn, suites, tc: TrainConfig, seed: int, eval_batches: int) -> Dict[str, Any]:
    seed_everything(seed)
    device = tc.device if tc.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = factory().to(device)
    model.train()
    objective = HtSB12Objective(margin=0.6, margin_weight=0.03, ratio_reg=1e-3, warmup_steps=tc.warmup_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    best_val = -1.0
    best_step = 0
    best_state = None
    prev_loss = None
    spikes = 0
    max_spike = 0.0
    final_loss = float("nan")
    train_log: List[Dict[str, Any]] = []

    for step in range(1, tc.steps + 1):
        lr = cosine_with_warmup(step, tc.steps, tc.warmup_steps, tc.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        b = train_fn(tc.batch_size, device, seed * 1_000_000 + step)
        out = forward_model(model, b)
        loss = compute_loss(model, out, b.labels, step, objective)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if tc.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        final_loss = float(loss.detach().cpu())
        if prev_loss is not None and prev_loss > 1e-8:
            ratio = final_loss / prev_loss
            if ratio > 1.5:
                spikes += 1
                max_spike = max(max_spike, ratio)
        prev_loss = final_loss

        if step % tc.eval_every == 0 or step == tc.steps:
            val_acc, val_loss = eval_accuracy_and_loss(model, val_fn, min(512, tc.batch_size * 2), device, seed * 2_000_000 + step * 100, eval_batches)
            if val_acc > best_val:
                best_val = val_acc
                best_step = step
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            train_log.append({
                "protocol": protocol,
                "model": model_name,
                "seed": seed,
                "step": step,
                "train_loss": round(final_loss, 4),
                "val_acc": round(val_acc * 100, 3),
                "val_loss": round(val_loss, 4),
                "lr": lr,
            })
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    suite_results: Dict[str, Any] = {}
    for name, fn in suites.items():
        eval_bs = tc.batch_size * 2 if tc.steps <= 5 else min(1024, max(512, tc.batch_size * 2))
        acc, loss = eval_accuracy_and_loss(model, fn, eval_bs, device, seed * 3_000_000 + len(name) * 1000, eval_batches)
        suite_results[f"{name}_acc"] = acc
        suite_results[f"{name}_loss"] = loss

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


def mean_std(vals):
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.min()), float(arr.max())


def summarize(runs, suite_names_by_protocol):
    rows=[]
    for p in sorted(set(r["protocol"] for r in runs)):
        suites=suite_names_by_protocol[p]
        pr=[r for r in runs if r["protocol"]==p]
        for m in sorted(set(r["model"] for r in pr)):
            mr=[r for r in pr if r["model"]==m]
            row={"protocol":p,"model":m,"params":mr[0]["params"],"n_seeds":len(mr)}
            row["val_mean"]=round(mean_std([r["best_val"] for r in mr])[0]*100,3)
            row["val_std"]=round(mean_std([r["best_val"] for r in mr])[1]*100,3)
            row["avg_spikes"]=round(float(np.mean([r["spikes"] for r in mr])),3)
            row["max_spike"]=round(float(max(r["max_spike"] for r in mr)),3)
            for s in suites:
                mu,sd,mn,mx=mean_std([r[f"{s}_acc"] for r in mr])
                lmu,lsd,_,_=mean_std([r[f"{s}_loss"] for r in mr])
                row[f"{s}_acc_mean"]=round(mu*100,3); row[f"{s}_acc_std"]=round(sd*100,3)
                row[f"{s}_acc_min"]=round(mn*100,3); row[f"{s}_acc_max"]=round(mx*100,3)
                row[f"{s}_loss_mean"]=round(lmu,5); row[f"{s}_loss_std"]=round(lsd,5)
            rows.append(row)
    return rows


def write_csv(path: Path, rows):
    if not rows: return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys and k != "train_log": keys.append(k)
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in keys})


def protocol_defs(max_eval_length: int, train_max_length: int, holdout_mod: int, num_classes: int):
    full_tasks=("length","count_a","count_b","count_digit")
    length_task=("length",); count_tasks=("count_a","count_b","count_digit")
    all_200=list(range(1,max_eval_length+1))
    extrap_train=list(range(1,train_max_length+1))
    extrap_test=list(range(train_max_length+1,max_eval_length+1))
    bucket_holdout=[x for x in all_200 if x % holdout_mod == 0]
    bucket_train=[x for x in all_200 if x % holdout_mod != 0]
    return {
        "d1_c3_extrapolation_fix": {
            "description": f"Train 1..{train_max_length}; test {train_max_length+1}..{max_eval_length}. Main OOD-fix target.",
            "train": batch_fn(extrap_train, full_tasks, "uniform", num_classes),
            "val": batch_fn(extrap_train, full_tasks, "uniform", num_classes),
            "suites": {
                "id_1_trainmax": batch_fn(extrap_train, full_tasks, "uniform", num_classes),
                "extrap_all": batch_fn(extrap_test, full_tasks, "uniform", num_classes),
                "extrap_length_only": batch_fn(extrap_test, length_task, "uniform", num_classes),
                "extrap_count_only": batch_fn(extrap_test, count_tasks, "uniform", num_classes),
                "extrap_biased_count": batch_fn(extrap_test, count_tasks, "biased_count", num_classes),
            },
        },
        "d2_bucket_interpolation_fix": {
            "description": f"Train 1..{max_eval_length} except length % {holdout_mod} == 0; test withheld buckets.",
            "train": batch_fn(bucket_train, full_tasks, "uniform", num_classes),
            "val": batch_fn(bucket_train, full_tasks, "uniform", num_classes),
            "suites": {
                "seen_buckets": batch_fn(bucket_train, full_tasks, "uniform", num_classes),
                "heldout_buckets": batch_fn(bucket_holdout, full_tasks, "uniform", num_classes),
                "heldout_length_only": batch_fn(bucket_holdout, length_task, "uniform", num_classes),
                "heldout_count_only": batch_fn(bucket_holdout, count_tasks, "uniform", num_classes),
            },
        },
        "d3_full_range_sanity": {
            "description": f"Train/test 1..{max_eval_length}. Sanity check that digit head does not destroy in-range performance.",
            "train": batch_fn(all_200, full_tasks, "uniform", num_classes),
            "val": batch_fn(all_200, full_tasks, "uniform", num_classes),
            "suites": {
                "full_id": batch_fn(all_200, full_tasks, "uniform", num_classes),
                "full_length_only": batch_fn(all_200, length_task, "uniform", num_classes),
                "full_count_only": batch_fn(all_200, count_tasks, "uniform", num_classes),
            },
        },
    }


def make_result_card(path: Path, summary, config):
    lines=["# HtS-B12 Benchmark-D OOD-Fix Result Card", "", "## Protocol", ""]
    for k,v in config.items(): lines.append(f"- {k}: `{v}`")
    lines += ["", "## Summary"]
    for p in sorted(set(r["protocol"] for r in summary)):
        rows=[r for r in summary if r["protocol"]==p]
        suites=[k[:-9] for k in rows[0] if k.endswith("_acc_mean")]
        lines.append(f"\n### {p}")
        header=["Model","Params","Val"]+[f"{s} Acc" for s in suites]+["Avg spikes"]
        lines.append("| "+" | ".join(header)+" |")
        lines.append("|"+"|".join(["---"]*len(header))+"|")
        for r in rows:
            row=[r["model"], f"{int(r['params']):,}", f"{float(r['val_mean']):.2f} ± {float(r['val_std']):.2f}%"]
            for s in suites:
                row.append(f"{float(r[f'{s}_acc_mean']):.2f} ± {float(r[f'{s}_acc_std']):.2f}%")
            row.append(str(r["avg_spikes"]))
            lines.append("| "+" | ".join(row)+" |")
    lines += ["", "## Interpretation", "- The key comparison is `HtS-DigitOOD` vs `HtS-B12-ClassHead` on `extrap_all`, `extrap_length_only`, and `extrap_count_only`.", "- A gain on extrapolation without training on 129..200 supports the hypothesis that dense class heads were blocking numeric OOD generalization.", "- If digit head improves length but not count, the next fix should target count accumulation rather than label parameterization."]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--protocols", type=str, default="d1,d2", help="d1,d2,d3 or full protocol names")
    ap.add_argument("--train-max-length", type=int, default=128)
    ap.add_argument("--max-eval-length", type=int, default=200)
    ap.add_argument("--num-classes", type=int, default=256)
    ap.add_argument("--holdout-mod", type=int, default=7)
    ap.add_argument("--model-filter", type=str, default="class,digit", help="class,digit")
    ap.add_argument("--tiny-model", action="store_true", help="Use a tiny config for CPU smoke tests only; not for publication numbers.")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    args=ap.parse_args()
    if args.quick:
        args.steps=1; args.batch_size=2; args.eval_batches=1; args.seeds="42"; args.protocols="d1"; args.max_eval_length=16; args.train_max_length=10; args.num_classes=max(args.num_classes,128); args.tiny_model=True
    if args.num_classes <= args.max_eval_length: raise SystemExit("--num-classes must exceed --max-eval-length")
    device=args.device if args.device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available(): raise SystemExit("CUDA not available")
    if args.quick and device == "cpu":
        # CPU smoke tests are for code validation only.  Limiting threads avoids
        # severe oversubscription on shared notebook CPUs.
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    print(f"Device: {device}")
    if device.startswith("cuda"):
        import os
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES','<not-set>')}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    seeds=[int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    proto_map={"d1":"d1_c3_extrapolation_fix","d2":"d2_bucket_interpolation_fix","d3":"d3_full_range_sanity"}
    protocols=[proto_map.get(p.strip().lower(), p.strip()) for p in args.protocols.split(',') if p.strip()]
    cfg=build_hts_config(args.max_eval_length, args.num_classes, tiny=args.tiny_model)
    filters={x.strip().lower() for x in args.model_filter.split(',') if x.strip()}
    models=[]
    if "class" in filters or "baseline" in filters or "all" in filters:
        models.append(("HtS-B12-ClassHead", lambda: HtSB12Classifier(cfg)))
    if "digit" in filters or "ood" in filters or "all" in filters:
        models.append(("HtS-DigitOOD", lambda: HtSB12DigitClassifier(cfg, max_digit_value=args.max_eval_length)))
    if not models: raise SystemExit("No model selected")
    print(f"Seeds: {seeds}")
    print(f"Protocols: {protocols}")
    print("Parameter counts:")
    for n,f in models: print(f"  {n:<24} {count_parameters(f()):,}")

    allp=protocol_defs(args.max_eval_length,args.train_max_length,args.holdout_mod,args.num_classes)
    suite_names={p:list(allp[p]["suites"].keys()) for p in protocols}
    tc=TrainConfig(steps=args.steps,batch_size=args.batch_size,lr=1e-3,weight_decay=0.01,warmup_steps=max(20,min(250,args.steps//20)),grad_clip=1.0,eval_every=max(50,args.steps//10),device=device,seed=42)
    runs=[]; t0=time.time()
    for p in protocols:
        meta=allp[p]
        print("\n"+"#"*90); print(f"PROTOCOL {p}: {meta['description']}"); print("#"*90)
        for seed in seeds:
            print("\n"+"="*80); print(f"{p} | Seed {seed}"); print("="*80)
            for name,factory in models:
                print(f"\nTraining {name}...")
                r=train_one(p,name,factory,meta["train"],meta["val"],meta["suites"],tc,seed,args.eval_batches)
                runs.append(r)
                first=list(meta["suites"].keys())[0]
                print(f"  {name:<24} val={r['best_val']*100:6.2f}% {first}={r[first+'_acc']*100:6.2f}% best_step={r['best_step']} spikes={r['spikes']} max_spike={r['max_spike']:.2f}x")
    summ=summarize(runs,suite_names)
    print("\n"+"="*90); print("PUBLICATION BENCHMARK-D OOD-FIX SUMMARY"); print("="*90)
    for r in summ:
        print(f"\n[{r['protocol']}] {r['model']} params={r['params']:,} seeds={r['n_seeds']}")
        for k in r:
            if k.endswith('_acc_mean'):
                s=k[:-9]
                print(f"  {s:<22} {r[k]:6.2f} ± {r[f'{s}_acc_std']:.2f}% loss={r[f'{s}_loss_mean']:.4f}")
        print(f"  avg_spikes={r['avg_spikes']} max_spike={r['max_spike']}")
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    flat=[]; train=[]
    for r in runs:
        flat.append({k:v for k,v in r.items() if k!='train_log'}); train.extend(r['train_log'])
    write_csv(out/'all_runs.csv',flat); write_csv(out/'summary.csv',summ); write_csv(out/'training_curves.csv',train)
    conf={"seeds":seeds,"steps":args.steps,"batch_size":args.batch_size,"eval_batches":args.eval_batches,"protocols":protocols,"train_max_length":args.train_max_length,"max_eval_length":args.max_eval_length,"num_classes":args.num_classes,"holdout_mod":args.holdout_mod,"device":device,"model_filter":args.model_filter,"tiny_model":args.tiny_model,"models":[n for n,_ in models]}
    (out/'config.json').write_text(json.dumps(conf,indent=2),encoding='utf-8')
    make_result_card(out/'result_card.md',summ,conf)
    print(f"\nSaved results to: {out.resolve()}")
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
