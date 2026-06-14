from __future__ import annotations
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .config import HtSConfig, TransformerConfig, TrainConfig
from .device import HtSDeviceManager
from .models import HtSTransformerClassifier, StaticTransformerClassifier, count_parameters
from .losses import cross_entropy_with_margin, hts_regularization_loss
from .diagnostics import accuracy, collect_diagnostics
from .data.synthetic_tasks import SyntheticTaskBatcher, FAMILIES, VOCAB_SIZE, OUTPUT_DIM, MAX_LEN


# Lazy imports for new benchmarks to avoid circular imports
def _get_multi_step_batcher():
    from .data.multi_step_reasoning import MultiStepBatcher
    return MultiStepBatcher()


def _get_compositional_batcher():
    from .data.compositional_tasks import CompositionalBatcher
    return CompositionalBatcher()


def _get_string_length_batcher():
    from .data.string_length_tasks import StringLengthBatcher
    return StringLengthBatcher()


BENCHMARK_DEFAULTS = {
    "synthetic": {"num_tasks": 30, "max_len": MAX_LEN},
    "multi_step": {"num_tasks": 110, "max_len": 18},
    "compositional": {"num_tasks": 256, "max_len": 12},
    "string_length": {"num_tasks": 10, "max_len": 200},
}


def set_seed(seed: int) -> None:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(kind: str = "hts", hts_config: Optional[HtSConfig] = None, tf_config: Optional[TransformerConfig] = None) -> nn.Module:
    kind = kind.lower()
    if kind in {"hts", "b12", "hts-b12"}:
        return HtSTransformerClassifier(hts_config or HtSConfig(vocab_size=VOCAB_SIZE, output_dim=OUTPUT_DIM, max_len=MAX_LEN))
    if kind in {"transformer", "static", "baseline"}:
        return StaticTransformerClassifier(tf_config or TransformerConfig(vocab_size=VOCAB_SIZE, output_dim=OUTPUT_DIM, max_len=MAX_LEN))
    raise ValueError(f"Unknown model kind: {kind}")


def build_batcher(benchmark: str):
    if benchmark == "synthetic":
        return SyntheticTaskBatcher()
    elif benchmark == "multi_step":
        return _get_multi_step_batcher()
    elif benchmark == "compositional":
        return _get_compositional_batcher()
    elif benchmark == "string_length":
        return _get_string_length_batcher()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(BENCHMARK_DEFAULTS)}")


def apply_benchmark_defaults(hts_cfg: HtSConfig, tf_cfg: TransformerConfig, benchmark: str):
    defaults = BENCHMARK_DEFAULTS.get(benchmark, {})
    if "num_tasks" in defaults:
        hts_cfg.num_tasks = defaults["num_tasks"]
        tf_cfg.num_tasks = defaults["num_tasks"]
    if "max_len" in defaults:
        hts_cfg.max_len = defaults["max_len"]
        tf_cfg.max_len = defaults["max_len"]
    return hts_cfg, tf_cfg


@torch.no_grad()
def evaluate(model: nn.Module, batcher, device, batch_size: int = 64, eval_batches: int = 10, benchmark: str = "synthetic") -> Dict[str, float]:
    model.eval()
    if benchmark == "synthetic":
        return _evaluate_synthetic(model, batcher, device, batch_size, eval_batches)
    elif benchmark == "multi_step":
        return _evaluate_multi_step(model, batcher, device, batch_size, eval_batches)
    elif benchmark == "compositional":
        return _evaluate_compositional(model, batcher, device, batch_size, eval_batches)
    elif benchmark == "string_length":
        return _evaluate_string_length(model, batcher, device, batch_size, eval_batches)
    else:
        return _evaluate_generic(model, batcher, device, batch_size, eval_batches)


def _evaluate_synthetic(model: nn.Module, batcher: SyntheticTaskBatcher, device, batch_size: int, eval_batches: int) -> Dict[str, float]:
    rows: Dict[str, List[float]] = {"macro_acc": [], "macro_loss": []}
    for family in FAMILIES:
        accs, losses = [], []
        for _ in range(eval_batches):
            batch = batcher.batch(batch_size, family=family, device=device)
            logits = model(batch["input_ids"], batch["task_ids"])
            loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
            accs.append(accuracy(logits, batch["labels"]))
            losses.append(float(loss.detach()))
        rows[f"{family}_acc"] = [float(np.mean(accs))]
        rows[f"{family}_loss"] = [float(np.mean(losses))]
        rows["macro_acc"].append(float(np.mean(accs)))
        rows["macro_loss"].append(float(np.mean(losses)))
    out = {k: v[0] if len(v) == 1 else float(np.mean(v)) for k, v in rows.items()}
    out.update(collect_diagnostics(model))
    model.train()
    return out


def _evaluate_multi_step(model: nn.Module, batcher, device, batch_size: int, eval_batches: int) -> Dict[str, float]:
    from .data.multi_step_reasoning import CHAIN_TEMPLATES, CHAIN_ID_OFFSET
    accs, losses = [], []
    for _ in range(eval_batches):
        batch = batcher.batch(batch_size, device=device)
        logits = model(batch["input_ids"], batch["task_ids"])
        loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
        accs.append(accuracy(logits, batch["labels"]))
        losses.append(float(loss.detach()))

    # Per-chain evaluation
    per_chain: Dict[str, List[float]] = {}
    for tmpl in CHAIN_TEMPLATES:
        chain_accs = []
        for _ in range(max(2, eval_batches // 2)):
            # Generate a batch with only this chain
            from .data.multi_step_reasoning import _tok, CHAIN_ID_OFFSET
            import random as _rand
            rows = []
            for _ in range(batch_size):
                args = [_rand.randint(0, 10) for _ in range(tmpl["n_args"])]
                result = tmpl["compute"](args)
                tokens = [1]  # CLS
                for i, a in enumerate(args):
                    tokens.append(_tok(a))
                    if i < len(args) - 1:
                        tokens.append(4)  # TOKEN_OP
                tokens.append(2)  # SEP
                tokens = tokens[:18] + [0] * (18 - len(tokens))
                rows.append((tokens, CHAIN_ID_OFFSET + tmpl["id"], int(result % 128)))
            inp = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)
            tid = torch.tensor([r[1] for r in rows], dtype=torch.long, device=device)
            lbl = torch.tensor([r[2] for r in rows], dtype=torch.long, device=device)
            logits = model(inp, tid)
            chain_accs.append(accuracy(logits, lbl))
        per_chain[tmpl["name"]] = [float(np.mean(chain_accs))]

    out = {
        "macro_acc": float(np.mean(accs)),
        "macro_loss": float(np.mean(losses)),
    }
    for name, vals in per_chain.items():
        out[f"{name}_acc"] = vals[0]
    out.update(collect_diagnostics(model))
    model.train()
    return out


def _evaluate_compositional(model: nn.Module, batcher, device, batch_size: int, eval_batches: int) -> Dict[str, float]:
    import random as _rand_local
    from .data.compositional_tasks import COMPOSITIONS, OP_NAMES, COMP_ID_OFFSET, _tok, TOKEN_THEN, NUM_OFFSET, PAD, CLS, SEP
    accs, losses = [], []
    for _ in range(eval_batches):
        batch = batcher.batch(batch_size, device=device)
        logits = model(batch["input_ids"], batch["task_ids"])
        loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
        accs.append(accuracy(logits, batch["labels"]))
        losses.append(float(loss.detach()))

    # Per-op1-group evaluation
    per_op1: Dict[str, List[float]] = {}
    for op1 in OP_NAMES:
        op1_accs = []
        comps = [c for c in COMPOSITIONS if c["op1"] == op1]
        for _ in range(max(2, eval_batches // 2)):
            rows = []
            for _ in range(batch_size):
                comp = _rand_local.choice(comps)
                a = _rand_local.randint(0, 10)
                b = _rand_local.randint(0, 10)
                c = _rand_local.randint(0, 10)
                intermediate = comp["fn1"](a, b)
                result = comp["fn2"](intermediate, c)
                op1_idx = OP_NAMES.index(comp["op1"])
                op2_idx = OP_NAMES.index(comp["op2"])
                tokens = [CLS, _tok(a), _tok(b), _tok(c), TOKEN_THEN, NUM_OFFSET + op1_idx, NUM_OFFSET + op2_idx, SEP]
                tokens = tokens[:12] + [PAD] * (12 - len(tokens))
                rows.append((tokens, COMP_ID_OFFSET + comp["id"], int(result % 128)))
            inp = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)
            tid = torch.tensor([r[1] for r in rows], dtype=torch.long, device=device)
            lbl = torch.tensor([r[2] for r in rows], dtype=torch.long, device=device)
            logits = model(inp, tid)
            op1_accs.append(accuracy(logits, lbl))
        per_op1[op1] = [float(np.mean(op1_accs))]

    out = {
        "macro_acc": float(np.mean(accs)),
        "macro_loss": float(np.mean(losses)),
    }
    for op_name, vals in per_op1.items():
        out[f"{op_name}_acc"] = vals[0]
    out.update(collect_diagnostics(model))
    model.train()
    return out


def _evaluate_string_length(model: nn.Module, batcher, device, batch_size: int, eval_batches: int) -> Dict[str, float]:
    from .data.string_length_tasks import TASKS, FAMILIES
    accs, losses = [], []
    for _ in range(eval_batches):
        batch = batcher.batch(batch_size, device=device)
        logits = model(batch["input_ids"], batch["task_ids"])
        loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
        accs.append(accuracy(logits, batch["labels"]))
        losses.append(float(loss.detach()))

    # Per-family evaluation
    per_family: Dict[str, List[float]] = {}
    for family, task_ids_list in FAMILIES.items():
        family_accs = []
        for _ in range(max(2, eval_batches // 2)):
            # Generate batch with tasks from this family only
            import random as _rand_local
            tasks = [_rand_local.choice(task_ids_list) for _ in range(batch_size)]
            x_list, y_list = [], []
            for task_id in tasks:
                from .data.string_length_tasks import _generate_string, _string_to_tokens, _label_for_task
                s = _generate_string(task_id, min_len=2, max_len=8)
                tokens = _string_to_tokens(s, batcher.max_len)
                label = _label_for_task(task_id, s)
                x_list.append(tokens)
                y_list.append(label)
            inp = torch.tensor(x_list, dtype=torch.long, device=device)
            tid = torch.tensor(tasks, dtype=torch.long, device=device)
            lbl = torch.tensor(y_list, dtype=torch.long, device=device)
            logits = model(inp, tid)
            family_accs.append(accuracy(logits, lbl))
        per_family[family] = [float(np.mean(family_accs))]

    out = {
        "macro_acc": float(np.mean(accs)),
        "macro_loss": float(np.mean(losses)),
    }
    for family_name, vals in per_family.items():
        out[f"{family_name}_acc"] = vals[0]
    out.update(collect_diagnostics(model))
    model.train()
    return out


def _evaluate_generic(model: nn.Module, batcher, device, batch_size: int, eval_batches: int) -> Dict[str, float]:
    accs, losses = [], []
    for _ in range(eval_batches):
        batch = batcher.batch(batch_size, device=device)
        logits = model(batch["input_ids"], batch["task_ids"])
        loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
        accs.append(accuracy(logits, batch["labels"]))
        losses.append(float(loss.detach()))
    out = {
        "macro_acc": float(np.mean(accs)),
        "macro_loss": float(np.mean(losses)),
    }
    out.update(collect_diagnostics(model))
    model.train()
    return out


def train_synthetic(
    model_kind: str = "hts",
    train_config: Optional[TrainConfig] = None,
    hts_config: Optional[HtSConfig] = None,
    tf_config: Optional[TransformerConfig] = None,
    out_dir: str | Path = "runs/hts_synthetic",
) -> Dict[str, object]:
    cfg = train_config or TrainConfig()
    set_seed(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manager = HtSDeviceManager(cfg.device)

    # Build batcher and apply benchmark defaults
    batcher = build_batcher(cfg.benchmark)
    hts_cfg = hts_config or HtSConfig()
    tf_cfg = tf_config or TransformerConfig()
    hts_cfg, tf_cfg = apply_benchmark_defaults(hts_cfg, tf_cfg, cfg.benchmark)

    model = build_model(model_kind, hts_cfg, tf_cfg)
    model = manager.to_device(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    history: List[Dict[str, float]] = []

    for step in tqdm(range(1, cfg.steps + 1), desc=f"train-{model_kind}-{cfg.benchmark}-{manager.backend}"):
        model.train()
        batch = batcher.batch(cfg.batch_size, device=manager.device)
        logits = model(batch["input_ids"], batch["task_ids"])
        task_loss, ce, margin_loss = cross_entropy_with_margin(logits, batch["labels"], cfg.margin_weight, cfg.margin)
        reg, reg_parts = hts_regularization_loss(model, cfg.budget_weight, cfg.binary_weight, cfg.ratio_weight, cfg.task_offset_weight)
        loss = task_loss + reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        manager.optimizer_step(opt)
        if step % cfg.eval_every == 0 or step == 1 or step == cfg.steps:
            ev = evaluate(model, batcher, manager.device, cfg.batch_size, cfg.eval_batches, benchmark=cfg.benchmark)
            row = {
                "step": step,
                "train_loss": float(loss.detach()),
                "train_ce": float(ce.detach()),
                "train_margin_loss": float(margin_loss.detach()),
                "params": count_parameters(model),
                "backend": manager.backend,
                **reg_parts,
                **ev,
            }
            history.append(row)

    import csv, json
    hist_path = out_dir / f"{model_kind}_metrics.csv"
    if history:
        with hist_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    meta = {
        "model_kind": model_kind,
        "params": count_parameters(model),
        "device": str(manager.info),
        "train_config": asdict(cfg),
    }
    with (out_dir / f"{model_kind}_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    torch.save(model.state_dict(), out_dir / f"{model_kind}_state_dict.pt")
    return {"history": history, "meta": meta, "metrics_path": str(hist_path), "model": model}
