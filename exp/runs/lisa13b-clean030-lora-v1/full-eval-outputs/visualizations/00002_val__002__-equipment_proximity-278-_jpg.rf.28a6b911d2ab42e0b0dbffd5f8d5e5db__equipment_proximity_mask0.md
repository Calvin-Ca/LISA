# Base / Tuned Sample Comparison

- Sample: `val__002__-equipment_proximity-278-_jpg.rf.28a6b911d2ab42e0b0dbffd5f8d5e5db__equipment_proximity.jpg`
- Category: `equipment_proximity`
- Prompt: 标出人员或设备距离过近、存在碰撞风险的区域。
- Base IoU: `0.0081` | Tuned IoU: `0.0000` | Delta: `-0.0081`
- COCO source boxes: `1` | COCO target boxes: `1` | LISA polygons: `61`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0002_val__002__-equipment_proximity-278-_jpg.rf.28a6b911d2ab42e0b0dbffd5f8d5e5db__equipment_proximity_coco_source.jpg) | ![](comparison_assets/0002_val__002__-equipment_proximity-278-_jpg.rf.28a6b911d2ab42e0b0dbffd5f8d5e5db__equipment_proximity_coco_target.jpg) | ![](comparison_assets/0002_val__002__-equipment_proximity-278-_jpg.rf.28a6b911d2ab42e0b0dbffd5f8d5e5db__equipment_proximity_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0002_val__002__-equipment_proximity-278-_jpg.rf.28a6b911d2ab42e0b0dbffd5f8d5e5db__equipment_proximity_base_pred.jpg) | ![](comparison_assets/0002_val__002__-equipment_proximity-278-_jpg.rf.28a6b911d2ab42e0b0dbffd5f8d5e5db__equipment_proximity_tuned_pred.jpg) |
