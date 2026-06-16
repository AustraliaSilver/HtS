# Publication Benchmark-D: HtS OOD-Fix

Benchmark-C showed that HtS-B12 is strong inside the observed length support but weak on data outside the training length range. Benchmark-D tests a minimal fix aimed at the output bottleneck.

## Core hypothesis

The original classifier uses one dense logit per target value. In strict extrapolation, training labels stop at 128 while evaluation labels include 129–200. Classes 129–200 therefore receive no direct positive supervision. This can block OOD numeric generalization even if the encoder has learned useful computation.

Benchmark-D adds `HtS-DigitOOD`, which keeps the HtS-B12 encoder and generated-computation FFN unchanged, but replaces the 256-way dense class head with a compositional digit head:

```text
hundreds = h(x, task)
tens     = t(x, task)
ones     = o(x, task)
y_hat    = 100 * argmax(hundreds) + 10 * argmax(tens) + argmax(ones)
```

The model is still trained only on the training length range. It does not train on held-out/extrapolation labels.

## Main comparison

Run `HtS-B12-ClassHead` vs `HtS-DigitOOD`.

The key metrics are:

- `d1_c3_extrapolation_fix/extrap_all`
- `d1_c3_extrapolation_fix/extrap_length_only`
- `d1_c3_extrapolation_fix/extrap_count_only`
- `d2_bucket_interpolation_fix/heldout_buckets`

## Kaggle T4x2 dual-GPU command

```python
!python /kaggle/working/HtS/benchmarks/kaggle_publication_d_dualgpu.py \
  --protocols d1,d2 \
  --steps 5000 \
  --seeds 42,123,777
```

Default GPU mapping:

- GPU 0: `HtS-B12-ClassHead`
- GPU 1: `HtS-DigitOOD`

## Smoke test

```python
!python /kaggle/working/HtS/benchmarks/kaggle_publication_d_oodfix.py --quick
```

If smoke testing on CPU, use one thread if the notebook runtime has slow CPU thread scheduling:

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
```

## Interpretation

Possible outcomes:

1. `HtS-DigitOOD` improves `extrap_length_only` strongly but not `extrap_count_only`.
   - The dense output class head was blocking length extrapolation.
   - Count accumulation remains the next bottleneck.

2. `HtS-DigitOOD` improves both length and count extrapolation.
   - The output parameterization was a major bottleneck.
   - This is a strong result for the paper.

3. `HtS-DigitOOD` does not improve extrapolation.
   - The failure is not mainly the output head.
   - Next fix should target encoder-level count/length accumulation, positional extrapolation, or router regularization.

4. `HtS-DigitOOD` improves OOD but hurts full-range ID badly.
   - The fix trades off ID accuracy for OOD generalization.
   - Need hybrid CE + digit head.

## Important limitation

Digit heads are appropriate for numeric targets such as length/count. They should be presented as an OOD numeric-target parameterization, not as a universal HtS improvement.

## CPU validation mode

Use `--quick --device cpu` for a tiny smoke test.  This mode automatically uses
`--tiny-model`; it validates code paths only and is not a publication benchmark.
