# LISA ReasonSeg Benchmark

- Model: `./LISA13B`
- Dataset: `./dataset` / `ReasonSeg|train`
- Samples: `415`
- Precision: `bf16`
- Font: `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`
- Mask threshold: `0.0`

| Metric | Value |
| --- | ---: |
| gIoU | 0.3432 |
| cIoU | 0.2938 |
| Mean Dice | 0.4163 |
| Mean Precision | 0.4069 |
| Mean Recall | 0.5148 |
| Seconds / sample | 0.22 |

## Worst Samples

| Image | IoU | Dice | Prompt |
| --- | ---: | ---: | --- |
| `./dataset/reason_seg/ReasonSeg/train/train__002__-equipment_proximity-122-_jpg.rf.b1840d22e4e43d2933a6de6bb083cc3e__equipment_proximity.jpg` | 0.0000 | 0.0000 | 图中哪些位置存在设备靠近人员的安全隐患?请分割出来。 |
| `./dataset/reason_seg/ReasonSeg/train/train__002__-equipment_proximity-14-_JPG.rf.c55abb62b6a6b01706efcba94056ff9c__helmet_missing.jpg` | 0.0000 | 0.0000 | 标出未按规定佩戴安全帽的作业人员。 |
| `./dataset/reason_seg/ReasonSeg/train/train__002__-equipment_proximity-162-_jpg.rf.ef4ba03daa6d2bb4477612a82d9f6225__equipment_proximity.jpg` | 0.0000 | 0.0000 | 标出人员或设备距离过近、存在碰撞风险的区域。 |
| `./dataset/reason_seg/ReasonSeg/train/train__002__-equipment_proximity-184-_jpg.rf.b37da60872d701b4a5f9ee453c0b2988__equipment_proximity.jpg` | 0.0000 | 0.0000 | 标出人员或设备距离过近、存在碰撞风险的区域。 |
| `./dataset/reason_seg/ReasonSeg/train/train__002__-equipment_proximity-201-_jpg.rf.fb6aebd232ec3047b6d5a85f4eabee05__equipment_proximity.jpg` | 0.0000 | 0.0000 | 标出人员或设备距离过近、存在碰撞风险的区域。 |

## Best Samples

| Image | IoU | Dice | Prompt |
| --- | ---: | ---: | --- |
| `./dataset/reason_seg/ReasonSeg/train/train__002__-opening_unprotected-50-_jpg.rf.9162ad14b8be4f1404b65037f3eccf0d__opening_unprotected.jpg` | 0.9809 | 0.9904 | 把缺少防护、可能导致坠落的开口区域分割出来。 |
| `./dataset/reason_seg/ReasonSeg/train/train__004__IMG20241113161227_jpg.rf.fb479937d3ffa5cd2260ce77d27baef5__no_jacket.jpg` | 0.9709 | 0.9852 | 标出未按要求穿戴反光背心的工人。 |
| `./dataset/reason_seg/ReasonSeg/train/train__004__IMG20241113161548_jpg.rf.0e98b481240e8114eb9bda9cb0b665dd__safe.jpg` | 0.9663 | 0.9829 | 圈出图中被标注为安全状态的作业人员或区域。 |
| `./dataset/reason_seg/ReasonSeg/train/train__004__IMG20241113161143_jpg.rf.90d1f31c75efe6a01f88aefa85ce9c0f__no_jacket.jpg` | 0.9647 | 0.9820 | 标出未按要求穿戴反光背心的工人。 |
| `./dataset/reason_seg/ReasonSeg/train/train__002__-opening_unprotected-69-_jpg.rf.09af43fcc74b43b5e039305fa80ddb55__opening_unprotected.jpg` | 0.9644 | 0.9819 | 把缺少防护、可能导致坠落的开口区域分割出来。 |
