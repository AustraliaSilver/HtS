#!/usr/bin/env bash
set -e
python -m hts.cli --model hts --device cpu --steps 300 --batch-size 64 --out-dir runs/hts_cpu
