# Dual-GPU Kaggle Benchmark Mode

This repository includes a dual-GPU launcher for Kaggle T4x2 notebooks.
It runs the two independent baselines concurrently:

- **HtS-B12** on physical GPU `0`
- **Transformer-ParamMatched** on physical GPU `1`

The launcher uses two separate Python processes with `CUDA_VISIBLE_DEVICES`.
Inside each subprocess, the selected physical GPU appears as local `cuda:0`, which is expected.

## Why process-based dual GPU?

HtS-B12 and Transformer are independent baseline runs.  They do not need model-parallel or data-parallel synchronization.  Running them as separate processes is simpler and faster than using `DataParallel` or `DistributedDataParallel` for this benchmark.

## Quick smoke test

```bash
python /kaggle/working/HtS/benchmarks/kaggle_publication_c_dualgpu.py --quick
```

## Run C1 only

```bash
python /kaggle/working/HtS/benchmarks/kaggle_publication_c_dualgpu.py \
  --protocols c1 \
  --steps 5000 \
  --seeds 42,123,777
```

## Run full C1+C2+C3 diagnostic suite

```bash
python /kaggle/working/HtS/benchmarks/kaggle_publication_c_dualgpu.py \
  --protocols c1,c2,c3 \
  --steps 5000 \
  --seeds 42,123,777
```

## Outputs

Merged results are saved to:

```text
publication_c_results_dualgpu/
├── all_runs.csv
├── summary.csv
├── training_curves.csv
├── config.json
└── result_card.md
```

Temporary per-model folders are also created unless removed:

```text
publication_c_results_dualgpu_hts_gpu0/
publication_c_results_dualgpu_tfm_gpu1/
```

## Important notes

1. Use a Kaggle notebook with **2× T4 GPUs** enabled.
2. If only one GPU is visible, the dual launcher exits with an error.
3. Do not compare wall-clock times from single-GPU sequential and dual-GPU parallel runs directly unless you document the execution mode.
4. Numerical results should remain independent because HtS and Transformer are trained in separate processes with separate CUDA visibility.
5. For ablation, run `--include-ablation`, but note that `HtS-NoSoft` is placed on GPU 0 and may share that GPU with HtS if run concurrently.

## Recommended publication workflow

1. Run `--quick` to confirm both GPUs are visible.
2. Run C1 first.
3. Run C2 if C1 is good.
4. Run C3 last because it is the hardest true extrapolation diagnostic.
5. Report `summary.csv` and `result_card.md` with the paper.
