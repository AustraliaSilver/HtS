import csv

def read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)

import sys
run = sys.argv[1] if len(sys.argv) > 1 else 'compare'
hts = read_csv(f'runs/hts_{run}/hts_metrics.csv')
tf = read_csv(f'runs/transformer_{run}/transformer_metrics.csv')

header = f"{'Step':>6} | {'HtS-B12 Acc':>12} | {'TF Acc':>12} | {'Delta':>12} | {'HtS Loss':>12} | {'TF Loss':>12}"
print('=' * 90)
print(header)
print('=' * 90)

for h, t in zip(hts, tf):
    step = h['step']
    h_acc = float(h['macro_acc'])
    t_acc = float(t['macro_acc'])
    delta = h_acc - t_acc
    h_loss = float(h['macro_loss'])
    t_loss = float(t['macro_loss'])
    print(f"{step:>6} | {h_acc:>11.4f} | {t_acc:>11.4f} | {delta:>+11.4f} | {h_loss:>12.4f} | {t_loss:>12.4f}")

print('=' * 90)

h_final = float(hts[-1]['macro_acc'])
t_final = float(tf[-1]['macro_acc'])
print(f"\nFinal (step 300):")
print(f"  HtS-B12:        {h_final:.4f} ({h_final*100:.2f}%)")
print(f"  Transformer:    {t_final:.4f} ({t_final*100:.2f}%)")
print(f"  Delta:          {h_final - t_final:+.4f} ({(h_final - t_final)*100:+.2f}%)")
print(f"  HtS params:     {hts[-1]['params']}")
print(f"  TF params:      {tf[-1]['params']}")
print(f"  Param overhead: {int(hts[-1]['params']) - int(tf[-1]['params'])} ({((int(hts[-1]['params']) / int(tf[-1]['params'])) - 1)*100:+.1f}%)")

print(f"\nPer-family accuracy at step 300:")
for family in ['arith8', 'seq6', 'comp3', 'arith_ood']:
    h_acc = float(hts[-1][f'{family}_acc'])
    t_acc = float(tf[-1][f'{family}_acc'])
    print(f"  {family:>12}: HtS={h_acc:.4f}  TF={t_acc:.4f}  delta={h_acc-t_acc:+.4f}")

print(f"\nHtS-B12 diagnostics at step 300:")
print(f"  gate_main:     {float(hts[-1]['block0.hts_l0_gate_main']):.4f}")
print(f"  alpha_main:    {float(hts[-1]['block0.hts_l0_alpha_main']):.4f}")
print(f"  target1:       {float(hts[-1]['block0.hts_l0_target1']):.4f}")
print(f"  target2:       {float(hts[-1]['block0.hts_l0_target2']):.4f}")
print(f"  delta/base:    {float(hts[-1]['block0.hts_l0_delta_base_ratio']):.4f}")
print(f"  corr_ratio:    {float(hts[-1]['block0.hts_l0_corr_ratio']):.4f}")
print(f"  budget:        {float(hts[-1]['block0.hts_l0_budget']):.4f}")
