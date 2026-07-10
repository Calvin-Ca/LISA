#!/usr/bin/env bash
set -euo pipefail

# Remote Linux GPU server.
# Fill the exact local checkpoint path used in the completed run.
python benchmark_reason_seg.py \
  --version /path/to/lisa13b-local-checkpoint \
  --dataset_dir ./dataset \
  --val_dataset "ReasonSeg|val" \
  --output_dir ./benchmark_outputs/lisa13b-local-val \
  --precision bf16 \
  --save_visualizations

