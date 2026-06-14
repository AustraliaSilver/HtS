#!/usr/bin/env bash
set -e
# Requires torch-xla installed in a TPU runtime.
python -m hts.cli --model hts --device tpu --steps 2000 --batch-size 256 --eval-every 200 --out-dir runs/hts_tpu
