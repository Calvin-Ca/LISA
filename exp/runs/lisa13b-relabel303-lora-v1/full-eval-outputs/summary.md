# LISA ReasonSeg Benchmark

- Model: `./runs/lisa13b-relabel303-lora-v1/merged_hf`
- Dataset: `./dataset` / `ReasonSeg|val`
- Samples: `86`
- Precision: `bf16`
- Font: `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`
- Mask threshold: `0.0`

| Metric | Value |
| --- | ---: |
| gIoU | 0.4112 |
| cIoU | 0.3263 |
| Mean Dice | 0.4752 |
| Mean Precision | 0.5102 |
| Mean Recall | 0.4883 |
| Seconds / sample | 0.42 |

## Worst Samples

| Image | IoU | Dice | Prompt |
| --- | ---: | ---: | --- |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-guardrail_missing-14-_jpg.rf.317f592ca167baf04b16296d87650912__guardrail_missing.jpg` | 0.0000 | 0.0000 | 指出没有设置防护栏杆的临边区域。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-opening_unprotected-26-_jpg.rf.8a97f7db61ee9371c7a662776cd884b2__opening_unprotected.jpg` | 0.0000 | 0.0000 | 圈出没有防护的洞口或临边区域。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-poor_housekeeping-132-_jpg.rf.c6a2c42ceb4d1eae5cd2307fedce5fe6__poor_housekeeping.jpg` | 0.0000 | 0.0000 | 图中哪些区域存在场地整理不到位的安全隐患?请分割出来。 |
| `./dataset/reason_seg/ReasonSeg/val/val__004__IMG20241113160658_jpg.rf.509371336267fc48e02387bc78859d8f__no_helmet.jpg` | 0.0000 | 0.0000 | 标出未按规定佩戴安全帽的作业人员。 |
| `./dataset/reason_seg/ReasonSeg/val/val__004__IMG20241113160658_jpg.rf.509371336267fc48e02387bc78859d8f__no_jacket.jpg` | 0.0000 | 0.0000 | 标出未按要求穿戴反光背心的工人。 |

## Best Samples

| Image | IoU | Dice | Prompt |
| --- | ---: | ---: | --- |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-poor_housekeeping-93-_jpg.rf.893e8f8298279e8c46d0fa8a503d58ff__poor_housekeeping.jpg` | 0.9696 | 0.9846 | 圈出施工现场杂乱、可能影响通行安全的位置。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-helmet_missing-238-_jpg.rf.d793369e91ce496d9e82d59c120a433a__helmet_missing.jpg` | 0.9635 | 0.9814 | 现场哪些人员存在未戴安全帽的安全隐患?请分割出来。 |
| `./dataset/reason_seg/ReasonSeg/val/val__004__IMG_20241113_161953162_jpg.rf.2fc5157cdd82110c6541270dcb7e5f82__safe.jpg` | 0.9556 | 0.9773 | 标出现场处于安全状态的目标。 |
| `./dataset/reason_seg/ReasonSeg/val/val__004__IMG_20241113_161954_jpg.rf.6d1a5633986ccbdc7de79340bce250ae__safe.jpg` | 0.9507 | 0.9747 | 圈出图中被标注为安全状态的作业人员或区域。 |
| `./dataset/reason_seg/ReasonSeg/val/val__004__IMG_20241113_162132082_jpg.rf.391adeed04f63ff7aa322dc4173f4a3a__no_helmet.jpg` | 0.9494 | 0.9740 | 标出未按规定佩戴安全帽的作业人员。 |
