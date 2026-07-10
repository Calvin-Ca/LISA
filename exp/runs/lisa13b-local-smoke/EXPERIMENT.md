# lisa13b-local-smoke

## Background

Smoke test for the local LISA-13B benchmark pipeline. This run is only used to verify that the model checkpoint, vision tower, dataset path, CUDA environment, and mask output path are connected correctly.

Do not use this run as a formal metric source.

## Configuration

- Model: LISA-13B local checkpoint
- Checkpoint: to fill
- Dataset: ReasonSeg
- Split: likely `ReasonSeg|val`
- Max samples: small subset, to fill
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
- `outputs/samples_by_iou.md`
- `outputs/visualizations/`
- `outputs/pred_masks/`

## Metrics

- Samples: to fill from `outputs/summary.json`
- gIoU: not for reporting
- cIoU: not for reporting
- Mean Dice: not for reporting
- Mean Precision: not for reporting
- Mean Recall: not for reporting

## Conclusion

Pipeline sanity check. Use `lisa13b-local-val` for reportable validation metrics.

## Notes

- Import existing remote output with: `cp -a benchmark_outputs/lisa13b-local-smoke/. exp/runs/lisa13b-local-smoke/outputs/`

