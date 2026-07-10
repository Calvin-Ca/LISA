# LISA ReasonSeg 评测

- 模型: `./LISA13B`
- 数据集: `./dataset` / `ReasonSeg|val`
- 样本数: `3`
- 精度: `bf16`
- 掩码阈值: `0.0`

| 指标 | 数值 |
| --- | ---: |
| gIoU | 0.2611 |
| cIoU | 0.3104 |
| 平均 Dice | 0.3544 |
| 平均精确率 | 0.3631 |
| 平均召回率 | 0.6200 |
| 单样本耗时(秒) | 0.29 |

## 最差样本

| 图像 | IoU | Dice | 指令 |
| --- | ---: | ---: | --- |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-equipment_proximity-278-_jpg.rf.28a6b911d2ab42e0b0dbffd5f8d5e5db__equipment_proximity.jpg` | 0.0081 | 0.0160 | 标出人员或设备距离过近、存在碰撞风险的区域。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-equipment_proximity-114-_jpg.rf.07492924a04683042df5548fae45af02__equipment_proximity.jpg` | 0.1739 | 0.2963 | 圈出施工现场设备邻近作业人员的危险区域。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-equipment_proximity-203-_jpg.rf.f10bae0078f7b417b8da3c3f67acc70f__equipment_proximity.jpg` | 0.6013 | 0.7510 | 图中哪些位置存在设备靠近人员的安全隐患?请分割出来。 |

## 最佳样本

| 图像 | IoU | Dice | 指令 |
| --- | ---: | ---: | --- |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-equipment_proximity-203-_jpg.rf.f10bae0078f7b417b8da3c3f67acc70f__equipment_proximity.jpg` | 0.6013 | 0.7510 | 图中哪些位置存在设备靠近人员的安全隐患?请分割出来。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-equipment_proximity-114-_jpg.rf.07492924a04683042df5548fae45af02__equipment_proximity.jpg` | 0.1739 | 0.2963 | 圈出施工现场设备邻近作业人员的危险区域。 |
| `./dataset/reason_seg/ReasonSeg/val/val__002__-equipment_proximity-278-_jpg.rf.28a6b911d2ab42e0b0dbffd5f8d5e5db__equipment_proximity.jpg` | 0.0081 | 0.0160 | 标出人员或设备距离过近、存在碰撞风险的区域。 |
