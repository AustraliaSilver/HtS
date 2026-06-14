# HtS-B12 Architecture Notes

## Core block

The HtS-B12 FFN computes:

```text
base1 = Linear1(X)
main1 = gate * alpha * ratio_normalize(GeneratedMain1(X,t), base1, target1)
corr1 = corr_gate * corr_alpha * GeneratedCorr1(X,t)
h = GELU(base1 + main1 + corr1)

base2 = Linear2(h)
main2 = gate * alpha * ratio_normalize(GeneratedMain2(h,t), base2, target2)
Y = base2 + main2
```

## Generated diagonal low-rank map

```text
Generated(X,t) = (X A^T) diag(m(t) * (s_task(t) + s_tune(X,t))) B^T
```

- `A` and `B` are hard trainable basis matrices.
- `m(t)` is an adaptive rank mask.
- `s_task(t)` is task-level soft coefficient.
- `s_tune(X,t)` is input-conditioned tuning.

## Why this is the selected endpoint

Earlier variants failed for specific reasons:

- B5/B6 deliberation: soft delta was near zero.
- B7 true FFN update: first variant where generated delta had real delta/base ratio.
- B9 ratio-router: best loss/calibration.
- B10 optimized: preserved most performance with fewer parameters.
- B11: hybrid improved loss but not robust accuracy.
- B12: task-specific router offsets + margin loss gave best CPU-scale accuracy.

## Device abstraction

Device logic is isolated in `hts/device.py`. TPU is optional and depends on PyTorch/XLA.
