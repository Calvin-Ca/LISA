# LISA ReasonSeg Benchmark

- Model: `./LISA13B`
- Dataset: `./dataset` / `ReasonSeg|val`
- Samples: `86`
- Precision: `bf16`
- Mask threshold: `0.0`

| Metric | Value |
| --- | ---: |
| gIoU | 0.3408 |
| cIoU | 0.3177 |
| Mean Dice | 0.4180 |
| Mean Precision | 0.4071 |
| Mean Recall | 0.5132 |
| Seconds / sample | 0.20 |

## Worst Samples

| Image | IoU | Dice | Prompt |
| --- | ---: | ---: | --- |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-guardrail_missing-14-_jpg.rf.317f592ca167baf04b16296d87650912__guardrail_missing.jpg` | 0.0000 | 0.0000 | 指出没有设置防护栏杆的临边区域。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-guardrail_missing-58-_jpg.rf.391f56af8166b7a9978f49176685f505__guardrail_missing.jpg` | 0.0000 | 0.0000 | 把缺少栏杆防护、存在坠落风险的部位分割出来。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-guardrail_missing-61-_jpg.rf.c0dd9f521c735b672d1318895a1aaffa__guardrail_missing.jpg` | 0.0000 | 0.0000 | 指出没有设置防护栏杆的临边区域。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-opening_unprotected-26-_jpg.rf.8a97f7db61ee9371c7a662776cd884b2__opening_unprotected.jpg` | 0.0000 | 0.0000 | 圈出没有防护的洞口或临边区域。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-poor_housekeeping-93-_jpg.rf.893e8f8298279e8c46d0fa8a503d58ff__poor_housekeeping.jpg` | 0.0000 | 0.0000 | 圈出施工现场杂乱、可能影响通行安全的位置。 |

## Best Samples

| Image | IoU | Dice | Prompt |
| --- | ---: | ---: | --- |
| `./dataset/reason_seg/ReasonSeg/val/val__004__IMG20241113160936_jpg.rf.162c6c777d07be2ca5c077d6906547f6__no_helmet.jpg` | 0.9569 | 0.9780 | 把没有做好头部防护、未戴安全帽的人分割出来。 |
| `./dataset/reason_seg/ReasonSeg/val/val__004__IMG_20241113_161953162_jpg.rf.2fc5157cdd82110c6541270dcb7e5f82__safe.jpg` | 0.9378 | 0.9679 | 标出现场处于安全状态的目标。 |
| `./dataset/reason_seg/ReasonSeg/val/val__004__IMG_20241113_161952050_jpg.rf.7c0b52ecb477f8f7ef8b71489dac19cf__safe.jpg` | 0.9341 | 0.9659 | 标出现场处于安全状态的目标。 |
| `./dataset/reason_seg/ReasonSeg/val/val__004__IMG_20241113_161954_jpg.rf.6d1a5633986ccbdc7de79340bce250ae__safe.jpg` | 0.9328 | 0.9652 | 圈出图中被标注为安全状态的作业人员或区域。 |
| `./dataset/reason_seg/ReasonSeg/val/val__004__IMG20241113161931_jpg.rf.a8d69015746e42c5014b2422eaf72c7b__safe.jpg` | 0.9296 | 0.9635 | 标出现场处于安全状态的目标。 |
