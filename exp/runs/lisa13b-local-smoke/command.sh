#!/usr/bin/env bash
set -euo pipefail

# Remote Linux GPU server.
# Fill the local checkpoint path and exact smoke-test sample count used.
python benchmark_reason_seg.py \
  --version /path/to/lisa13b-local-checkpoint \
  --dataset_dir ./dataset \
  --val_dataset "ReasonSeg|val" \
  --output_dir ./benchmark_outputs/lisa13b-local-smoke \
  --precision bf16 \
  --max_samples 5 \
  --save_visualizations

