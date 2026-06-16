# Publication Benchmark-C: Diagnostic Generalization Suite

Benchmark-B established a strong in-distribution result for HtS-B12 but poor held-out length extrapolation. Benchmark-C is a diagnostic suite designed to separate three possible explanations:

1. **Capacity / label availability**: the model may solve the task if trained across the full length range.
2. **Bucket interpolation**: the model may generalize to unseen length buckets if all positions up to the maximum are seen during training.
3. **True extrapolation**: the model may still fail when test lengths exceed the trained length range.

## Protocols

### C1: Seen full range
Train and test on lengths `1..max_eval_length`.

This asks whether HtS-B12 and the Transformer can solve the full label/range when examples from the entire range are available during training.

### C2: Bucket interpolation
Train on lengths `1..max_eval_length` except lengths divisible by `holdout_mod` and test on those withheld buckets.

This asks whether the model learns a smoother rule over lengths/counts, rather than memorizing observed length buckets.

### C3: Length extrapolation
Train on lengths `1..train_max_length` and test on `train_max_length+1..max_eval_length`.

This mirrors the hard extrapolation failure observed in Benchmark-B.

## Recommended Kaggle commands

Quick smoke:

```bash
python /kaggle/working/HtS/benchmarks/kaggle_publication_c.py --quick
```

Run C1 first:

```bash
python /kaggle/working/HtS/benchmarks/kaggle_publication_c.py \
  --protocols c1 --steps 5000 --seeds 42,123,777
```

Run full diagnostic suite:

```bash
python /kaggle/working/HtS/benchmarks/kaggle_publication_c.py \
  --protocols c1,c2,c3 --steps 5000 --seeds 42,123,777
```

## Interpretation

- If **C1 high, C3 low**, the model has capacity but lacks length extrapolation.
- If **C2 high, C3 low**, the model can interpolate within the seen positional range but cannot extrapolate beyond it.
- If **C2 low**, the model may be memorizing length buckets/classes rather than learning a smooth length/count rule.
- If biased/rare count suites remain low, count robustness is a separate weakness.

## Dual-GPU Kaggle mode

For Kaggle notebooks with 2× T4 GPUs, use:

```bash
python /kaggle/working/HtS/benchmarks/kaggle_publication_c_dualgpu.py \
  --protocols c1,c2,c3 \
  --steps 5000 \
  --seeds 42,123,777
```

This launches HtS-B12 on physical GPU 0 and Transformer-ParamMatched on physical GPU 1 using separate Python processes.  Merged results are saved in `publication_c_results_dualgpu/`.

See `docs/DUAL_GPU_KAGGLE.md` for details.
