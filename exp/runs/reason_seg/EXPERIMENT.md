# reason_seg

## Background

Legacy/default output directory from `benchmark_reason_seg.py`. The script default is `./benchmark_outputs/reason_seg`, so this run may contain whichever evaluation was executed without a custom `--output_dir`.

Confirm the actual model, split, and sample count from `outputs/summary.json` before using any metric.

## Configuration

- Model: to confirm from `outputs/summary.json`
- Checkpoint: to confirm
- Dataset: ReasonSeg
- Split: default is `ReasonSeg|val`, confirm from `outputs/summary.json`
- Max samples: to confirm
- Precision: to confirm
- Mask threshold: to confirm
- Save visualizations: to confirm
- Save masks: to confirm
- Device: remote Linux GPU server
- Date: to confirm

## Command

See `command.sh`.

## Outputs

- `outputs/summary.json`
- `outputs/summary.md`
- `outputs/per_sample_metrics.csv`
- `outputs/per_sample_metrics.jsonl`
- `outputs/per_sample_metrics_by_iou.csv`
- `outputs/samples_by_iou.md`
- `outputs/visualizations/`
- `outputs/pred_masks/`

## Metrics

- Samples: to fill from `outputs/summary.json`
- gIoU: to fill
- cIoU: to fill
- Mean Dice: to fill
- Mean Precision: to fill
- Mean Recall: to fill

## Conclusion

Treat as an imported legacy run until its exact configuration is confirmed.

## Notes

- Import existing remote output with: `cp -a benchmark_outputs/reason_seg/. exp/runs/reason_seg/outputs/`

