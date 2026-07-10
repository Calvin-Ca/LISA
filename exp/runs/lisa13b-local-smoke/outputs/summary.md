# LISA ReasonSeg Benchmark

- Model: `./LISA13B`
- Dataset: `./dataset` / `ReasonSeg|val`
- Samples: `3`
- Precision: `bf16`
- Mask threshold: `0.0`

| Metric | Value |
| --- | ---: |
| gIoU | 0.2611 |
| cIoU | 0.3104 |
| Mean Dice | 0.3544 |
| Mean Precision | 0.3631 |
| Mean Recall | 0.6200 |
| Seconds / sample | 0.29 |

## Worst Samples

| Image | IoU | Dice | Prompt |
| --- | ---: | ---: | --- |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-equipment_proximity-278-_jpg.rf.28a6b911d2ab42e0b0dbffd5f8d5e5db__equipment_proximity.jpg` | 0.0081 | 0.0160 | 标出人员或设备距离过近、存在碰撞风险的区域。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-equipment_proximity-114-_jpg.rf.07492924a04683042df5548fae45af02__equipment_proximity.jpg` | 0.1739 | 0.2963 | 圈出施工现场设备邻近作业人员的危险区域。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-equipment_proximity-203-_jpg.rf.f10bae0078f7b417b8da3c3f67acc70f__equipment_proximity.jpg` | 0.6013 | 0.7510 | 图中哪些位置存在设备靠近人员的安全隐患?请分割出来。 |

## Best Samples

| Image | IoU | Dice | Prompt |
| --- | ---: | ---: | --- |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-equipment_proximity-203-_jpg.rf.f10bae0078f7b417b8da3c3f67acc70f__equipment_proximity.jpg` | 0.6013 | 0.7510 | 图中哪些位置存在设备靠近人员的安全隐患?请分割出来。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-equipment_proximity-114-_jpg.rf.07492924a04683042df5548fae45af02__equipment_proximity.jpg` | 0.1739 | 0.2963 | 圈出施工现场设备邻近作业人员的危险区域。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-equipment_proximity-278-_jpg.rf.28a6b911d2ab42e0b0dbffd5f8d5e5db__equipment_proximity.jpg` | 0.0081 | 0.0160 | 标出人员或设备距离过近、存在碰撞风险的区域。 |
