# lisa13b-local-train

## Background

Evaluation of the local LISA-13B checkpoint on the training split. This run is used to inspect fitting behavior and catch annotation/model issues on samples seen during LoRA training or dataset construction.

Do not treat this as the primary generalization metric.

## Configuration

- Model: LISA-13B local checkpoint
- Checkpoint: to fill
- Dataset: ReasonSeg
- Split: `ReasonSeg|train`
- Max samples: all unless changed
- Precision: likely `bf16`
- Mask threshold: `0.0` unless changed
- Save visualizations: to fill
- Save masks: to fill
- Device: remote Linux GPU server
- Date: to fill

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

Use this run to compare against validation performance and identify overfitting or noisy labels.

## Notes

- Import existing remote output with: `cp -a benchmark_outputs/lisa13b-local-train/. exp/runs/lisa13b-local-train/outputs/`

