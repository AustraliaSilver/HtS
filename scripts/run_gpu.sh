#!/usr/bin/env bash
set -e
python -m hts.cli --model hts --device cuda --steps 2000 --batch-size 256 --eval-every 200 --out-dir runs/hts_cuda
