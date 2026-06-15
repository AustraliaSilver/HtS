# HtS-B12 Architecture Notes

HtS-B12 is a Hard-to-Soft generated-computation architecture. The central idea is not merely to add a new classifier head. Instead, HtS generates **soft weight updates inside the actual FFN computation path**.

## Core update

For each FFN block:

```text
base1 = W1(x)
main1 = gate(x,t) * alpha(x,t) * targeted_delta_1(x,t)
corr1 = corr_gate(x,t) * corr_alpha(x,t) * free_correction_1(x,t)
h = GELU(base1 + main1 + corr1)

base2 = W2(h)
main2 = gate(x,t) * alpha(x,t) * targeted_delta_2(h,t)
y = base2 + main2
```

## What B12 adds

B12 adds task-specific router offsets and margin-oriented training:

```text
router = shared_router(mean(x), task_embedding)
offset = task_offset_scale * tanh(task_router_offset[task])
router = router + offset
```

This lets each task bias gate/alpha/target-ratio while still sharing the same hard generator.

## Why this follows the original HtS philosophy

- Hard weights define the stable computation substrate.
- A hard router/generator produces soft, input/task-conditioned weight updates.
- Soft updates are trained through the final task loss.
- Generated computation happens inside the network, not only in an external head.

## Diagnostics

`model.hts_diagnostics()` returns values such as:

- `gate_main`
- `alpha_main`
- `gate_corr`
- `delta_base_ratio`
- `corr_ratio`
- `target1`, `target2`
- generated-rank coefficient statistics

These diagnostics are essential for detecting near-zero deltas or over-aggressive updates.
