# LISA ReasonSeg Benchmark

- Model: `./runs/lisa13b-clean030-lora-v1/merged_hf`
- Dataset: `./dataset` / `ReasonSegClean030|val`
- Samples: `42`
- Precision: `bf16`
- Font: `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`
- Mask threshold: `0.0`

| Metric | Value |
| --- | ---: |
| gIoU | 0.7119 |
| cIoU | 0.6642 |
| Mean Dice | 0.7868 |
| Mean Precision | 0.8300 |
| Mean Recall | 0.7687 |
| Seconds / sample | 0.42 |

## Worst Samples

| Image | IoU | Dice | Prompt |
| --- | ---: | ---: | --- |
| `./dataset/reason_seg/ReasonSegClean030/val/val__002__-poor_housekeeping-132-_jpg.rf.c6a2c42ceb4d1eae5cd2307fedce5fe6__poor_housekeeping.jpg` | 0.0000 | 0.0000 | 图中哪些区域存在场地整理不到位的安全隐患?请分割出来。 |
| `./dataset/reason_seg/ReasonSegClean030/val/val__004__IMG20241113160817_jpg.rf.9c7cd48d2297fc816a026d117f0df99b__no_helmet.jpg` | 0.0000 | 0.0000 | 把没有做好头部防护、未戴安全帽的人分割出来。 |
| `./dataset/reason_seg/ReasonSegClean030/val/val__004__IMG20241113160658_jpg.rf.509371336267fc48e02387bc78859d8f__no_helmet.jpg` | 0.0948 | 0.1732 | 标出未按规定佩戴安全帽的作业人员。 |
| `./dataset/reason_seg/ReasonSegClean030/val/val__004__IMG20241113160529_jpg.rf.01cc514ff77c809eb7eb263d362448cd__no_jacket.jpg` | 0.1952 | 0.3266 | 圈出没有穿反光衣或安全背心的作业人员。 |
| `./dataset/reason_seg/ReasonSegClean030/val/val__004__IMG_20241113_161340_jpg.rf.d2e6016a68c883c9a28b149919ca2c5c__no_helmet.jpg` | 0.2271 | 0.3702 | 标出未按规定佩戴安全帽的作业人员。 |

## Best Samples

| Image | IoU | Dice | Prompt |
| --- | ---: | ---: | --- |
| `./dataset/reason_seg/ReasonSegClean030/val/val__004__IMG20241113161529_jpg.rf.e77d13689e9d29512af57438dfc09685__no_helmet.jpg` | 0.9759 | 0.9878 | 现场哪些人员存在未戴安全帽的安全隐患?请分割出来。 |
| `./dataset/reason_seg/ReasonSegClean030/val/val__002__-helmet_missing-238-_jpg.rf.d793369e91ce496d9e82d59c120a433a__helmet_missing.jpg` | 0.9602 | 0.9797 | 现场哪些人员存在未戴安全帽的安全隐患?请分割出来。 |
| `./dataset/reason_seg/ReasonSegClean030/val/val__004__IMG_20241113_161758877_HDR_jpg.rf.362f3e155a81c2679a7690b2ef961abb__no_helmet.jpg` | 0.9550 | 0.9770 | 标出未按规定佩戴安全帽的作业人员。 |
| `./dataset/reason_seg/ReasonSegClean030/val/val__004__IMG_20241113_161953162_jpg.rf.2fc5157cdd82110c6541270dcb7e5f82__safe.jpg` | 0.9539 | 0.9764 | 标出现场处于安全状态的目标。 |
| `./dataset/reason_seg/ReasonSegClean030/val/val__004__IMG_20241113_161954_jpg.rf.6d1a5633986ccbdc7de79340bce250ae__safe.jpg` | 0.9504 | 0.9746 | 圈出图中被标注为安全状态的作业人员或区域。 |
