# Publication Benchmark-B for HtS-B12

This benchmark is intended to produce a cleaner result suitable for a paper-style report.  It is stricter than the first Kaggle validation script.

## Why this benchmark exists

The first stability benchmark showed useful signal, but it still had three limitations:

1. The Transformer baseline had fewer parameters than HtS-B12.
2. Some earlier string/count settings used `num_classes=128` while `max_length=200`, which can silently clip length/count labels.
3. ID accuracy alone is not sufficient for a publishable claim.

Benchmark-B fixes these issues by default.

## What it measures

Benchmark-B trains each model once per seed and evaluates the best-validation checkpoint on multiple suites:

| Suite | Meaning |
|---|---|
| `id` | Same-distribution test: train/test length range both `1-train_max_length`. |
| `heldout_length` | Train on short strings, test on longer strings. |
| `length_only` | Length task only over the full evaluation range. |
| `count_only` | Count tasks only over the full evaluation range. |
| `biased_count` | Count tasks under shifted token distribution. |

## Models

Default models:

- `HtS-B12`
- `Transformer-ParamMatched`

Optional models:

- `Transformer-Small`
- `HtS-NoSoft` ablation

`HtS-NoSoft` disables generated soft updates by setting soft-update strengths to zero.  It keeps the same overall architecture and parameters, so it tests whether the generated soft computation is actually helping.

## Recommended Kaggle command

```python
!python /kaggle/working/HtS/benchmarks/kaggle_publication_b.py \
    --steps 5000 \
    --seeds 42,123,777
```

With ablation:

```python
!python /kaggle/working/HtS/benchmarks/kaggle_publication_b.py \
    --steps 5000 \
    --seeds 42,123,777 \
    --include-ablation
```

Quick smoke:

```python
!python /kaggle/working/HtS/benchmarks/kaggle_publication_b.py --quick
```

## Outputs

Results are written to `publication_b_results/`:

- `summary.csv`
- `all_runs.csv`
- `training_curves.csv`
- `config.json`
- `result_card.md`

## How to interpret

A strong publishable result requires:

1. HtS-B12 beats `Transformer-ParamMatched` on `id` accuracy.
2. HtS-B12 is competitive or better on `heldout_length`.
3. HtS-B12 has acceptable seed variance.
4. The `HtS-NoSoft` ablation is lower than full HtS-B12.
5. The report states parameter counts and does not claim general superiority beyond the tested suites.

