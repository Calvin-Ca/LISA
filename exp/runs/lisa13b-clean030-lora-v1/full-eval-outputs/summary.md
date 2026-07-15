# LISA ReasonSeg Benchmark

- Model: `./runs/lisa13b-clean030-lora-v1/merged_hf`
- Dataset: `./dataset` / `ReasonSeg|val`
- Samples: `86`
- Precision: `bf16`
- Font: `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`
- Mask threshold: `0.0`

| Metric | Value |
| --- | ---: |
| gIoU | 0.4494 |
| cIoU | 0.3858 |
| Mean Dice | 0.5156 |
| Mean Precision | 0.5332 |
| Mean Recall | 0.5416 |
| Seconds / sample | 0.42 |

## Worst Samples

| Image | IoU | Dice | Prompt |
| --- | ---: | ---: | --- |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-equipment_proximity-278-_jpg.rf.28a6b911d2ab42e0b0dbffd5f8d5e5db__equipment_proximity.jpg` | 0.0000 | 0.0000 | 标出人员或设备距离过近、存在碰撞风险的区域。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-guardrail_missing-14-_jpg.rf.317f592ca167baf04b16296d87650912__guardrail_missing.jpg` | 0.0000 | 0.0000 | 指出没有设置防护栏杆的临边区域。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-opening_unprotected-51-_jpg.rf.7db855a017ceaddec7cf0044a0a8b7e2__opening_unprotected.jpg` | 0.0000 | 0.0000 | 图中哪些洞口或临边没有做防护?请分割出来。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-poor_housekeeping-132-_jpg.rf.c6a2c42ceb4d1eae5cd2307fedce5fe6__poor_housekeeping.jpg` | 0.0000 | 0.0000 | 图中哪些区域存在场地整理不到位的安全隐患?请分割出来。 |
| `./dataset/reason_seg/ReasonSeg/val/val__004__IMG20241113160529_jpg.rf.01cc514ff77c809eb7eb263d362448cd__unsafe.jpg` | 0.0000 | 0.0000 | 请分割图中存在安全风险的目标区域。 |

## Best Samples

| Image | IoU | Dice | Prompt |
| --- | ---: | ---: | --- |
| `./dataset/reason_seg/ReasonSeg/val/val__004__IMG20241113161529_jpg.rf.e77d13689e9d29512af57438dfc09685__no_helmet.jpg` | 0.9759 | 0.9878 | 现场哪些人员存在未戴安全帽的安全隐患?请分割出来。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-helmet_missing-238-_jpg.rf.d793369e91ce496d9e82d59c120a433a__helmet_missing.jpg` | 0.9602 | 0.9797 | 现场哪些人员存在未戴安全帽的安全隐患?请分割出来。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-poor_housekeeping-93-_jpg.rf.893e8f8298279e8c46d0fa8a503d58ff__poor_housekeeping.jpg` | 0.9596 | 0.9794 | 圈出施工现场杂乱、可能影响通行安全的位置。 |
| `./dataset/reason_seg/ReasonSeg/val/val__004__IMG_20241113_161758877_HDR_jpg.rf.362f3e155a81c2679a7690b2ef961abb__no_helmet.jpg` | 0.9550 | 0.9770 | 标出未按规定佩戴安全帽的作业人员。 |
| `./dataset/reason_seg/ReasonSeg/val/val__004__IMG_20241113_161953162_jpg.rf.2fc5157cdd82110c6541270dcb7e5f82__safe.jpg` | 0.9539 | 0.9764 | 标出现场处于安全状态的目标。 |
