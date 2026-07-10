# Experiments

This directory is the experiment ledger for LISA construction-safety runs.

Keep each experiment self-contained:

- `EXPERIMENT.md`: background, model/data config, metrics, conclusion.
- `command.sh`: the exact command used on the remote Linux GPU server.
- `outputs/`: copied benchmark artifacts such as `summary.json`, `summary.md`, CSV/JSONL metrics, masks, and visualizations.

Image files under `outputs/` are intentionally ignored by git. Keep metric files and notes tracked; store large visualizations locally or on the remote server.

## Runs

| Run | Purpose | Model | Dataset split | Status | Key output |
|---|---|---|---|---|---|
| `lisa13b-local-smoke` | Smoke test for local LISA-13B inference/eval chain | LISA-13B local checkpoint | small subset | pending output import | `outputs/summary.json` |
| `lisa13b-local-train` | Evaluate on training split to inspect fitting | LISA-13B local checkpoint | `ReasonSeg|train` | pending output import | `outputs/summary.json` |
| `lisa13b-local-val` | Evaluate on validation split for reportable metrics | LISA-13B local checkpoint | `ReasonSeg|val` | pending output import | `outputs/summary.json` |
| `reason_seg` | Legacy/default `benchmark_reason_seg.py` output directory | to confirm from summary | to confirm from summary | pending output import | `outputs/summary.json` |

## Import Existing Remote Outputs

Remote execution only. Copy each existing benchmark directory into the matching run:

```bash
cp -a benchmark_outputs/lisa13b-local-smoke/. exp/runs/lisa13b-local-smoke/outputs/
cp -a benchmark_outputs/lisa13b-local-train/. exp/runs/lisa13b-local-train/outputs/
cp -a benchmark_outputs/lisa13b-local-val/. exp/runs/lisa13b-local-val/outputs/
cp -a benchmark_outputs/reason_seg/. exp/runs/reason_seg/outputs/
```

After importing, update each `EXPERIMENT.md` with the actual command, sample count, and metrics from `outputs/summary.json`.

