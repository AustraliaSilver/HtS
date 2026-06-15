"""Quick diagnostic: HtS-B12 diagnostics on small model."""
import torch, time
from hts_b12 import HtSB12Classifier, HtSB12Config, TransformerClassifier, accuracy
from hts_b12.data.string_tasks import make_string_count_batch
from hts_b12.losses import HtSB12Objective

cfg = HtSB12Config(
    vocab_size=128, max_length=40, num_tasks=4, num_classes=40,
    d_model=32, n_heads=2, num_layers=1, dim_ff=64,
    task_dim=8, rank_main=4, rank_corr=2, dropout=0.0,
)

for name, MC in [("HtS-B12", HtSB12Classifier), ("Transformer", TransformerClassifier)]:
    model = MC(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    obj = HtSB12Objective(margin=0.6, margin_weight=0.03, ratio_reg=1e-3, warmup_steps=100)
    model.train()
    print(f"\n=== {name} ===")
    t0 = time.time()
    for step in range(1, 501):
        lr = 1e-3 * min(1.0, step / 100) * 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * min(1.0, (step - 100) / (500 - 100)))).item())
        for g in opt.param_groups:
            g['lr'] = lr
        batch = make_string_count_batch(128, 40, 'cpu', num_classes=40, seed=step)
        logits = model(batch.input_ids, batch.task_ids)
        loss_bd = obj(model, logits, batch.labels, step=step)
        opt.zero_grad()
        loss_bd.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                tr_acc = float(accuracy(logits, batch.labels))
                vb = make_string_count_batch(128, 40, 'cpu', num_classes=40, seed=step+10000)
                vl = model(vb.input_ids, vb.task_ids)
                va = float(accuracy(vl, vb.labels))
            s = loss_bd.scalars()
            extra = ""
            if name == "HtS-B12":
                d = model.hts_diagnostics()
                extra = f"  ratio={d['l0_layer0_b12_delta_base_ratio']:.3f} gate={d['l0_layer0_b12_gate_main']:.3f}"
            print(f"  step {step:3d} | loss {s['loss']:.4f} | train {tr_acc*100:.1f}% | val {va*100:.1f}%{extra}")
            model.train()
    t1 = time.time()
    print(f"  Time: {t1-t0:.1f}s")
