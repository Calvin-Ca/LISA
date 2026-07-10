#!/usr/bin/env bash
set -euo pipefail

# Remote Linux GPU server.
# This matches benchmark_reason_seg.py defaults except for explicit output_dir.
python benchmark_reason_seg.py \
  --version xinlai/LISA-13B-llama2-v1 \
  --dataset_dir ./dataset \
  --val_dataset "ReasonSeg|val" \
  --output_dir ./benchmark_outputs/reason_seg \
  --precision bf16

