# HtS-B12 Kaggle Stability Fix Notes

This package fixes the immediate instability/benchmark-control issues observed in `benchmarks/kaggle_validation_a.py`.

## Main diagnosis

The first Kaggle script reported a very poor HtS-B12 run for seed `123` while seed `42` was strong. The most important benchmark bug was:

```python
r = train_one("HtS-B12", HtSB12Classifier(cfg), bf, bf, tc, seed)
```

The model was constructed before `seed_everything(seed)` was called inside `train_one`. Therefore, the per-seed runs did **not** actually control model initialization. A bad HtS initialization could appear under a seed label, making the seed analysis misleading.

There was also a practical Kaggle issue: running

```python
!python /kaggle/working/HtS/benchmarks/kaggle_validation_a.py
```

may import an already-installed `hts_b12` from `site-packages`, not the freshly cloned repo source. The benchmark now inserts `repo/src` into `sys.path` before importing `hts_b12`.

## Fixes applied

1. **Seed before model creation**
   - `train_one` now receives a `model_factory` and constructs the model only after `seed_everything(seed)`.

2. **Use freshly cloned local source**
   - `benchmarks/kaggle_validation_a.py` prepends `ROOT/src` to `sys.path`.

3. **Pass attention masks**
   - Training/eval now call `model(input_ids, task_ids, attention_mask)`.

4. **Best-validation checkpoint**
   - Final test is evaluated from the best validation checkpoint, not necessarily the final training step.

5. **Vectorized data generator**
   - `src/hts_b12/data/string_tasks.py` no longer loops per sample, making Kaggle/GPU runs much faster.

6. **HtS stability patch**
   - HtS receives a direct task input embedding, so attention/CLS pooling is task-aware before the FFN routers.
   - HtS FFN router context uses padding-aware masked means.
   - Ratio-targeting has a safety clamp to avoid rare near-zero raw-delta overscaling.
   - B12 benchmark config uses slightly safer alpha/target/correction values.

7. **A2 print bug fixed**
   - HtS and Transformer A2 results are printed from separate variables.

## Recommended Kaggle usage

Fresh clone/uploaded repo:

```python
!python /kaggle/working/HtS/benchmarks/kaggle_validation_a.py --quick
```

Full validation:

```python
!python /kaggle/working/HtS/benchmarks/kaggle_validation_a.py
```

Custom shorter run:

```python
!python /kaggle/working/HtS/benchmarks/kaggle_validation_a.py --steps 2000 --seeds 42,123,777
```

If you want to ensure the local source is used, the benchmark prints no explicit path by default, but internally prepends `/kaggle/working/HtS/src` to `sys.path`.

## Important scientific note

This patch fixes benchmark-control and stability issues. It does **not** guarantee that HtS-B12 will always beat Transformer on every seed/split. If the fixed benchmark shows Transformer winning, that result should be accepted and used to improve HtS rather than hidden.
