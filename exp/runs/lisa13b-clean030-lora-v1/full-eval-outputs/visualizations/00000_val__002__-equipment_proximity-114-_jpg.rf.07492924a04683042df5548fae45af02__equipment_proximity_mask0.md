# Base / Tuned Sample Comparison

- Sample: `val__002__-equipment_proximity-114-_jpg.rf.07492924a04683042df5548fae45af02__equipment_proximity.jpg`
- Category: `equipment_proximity`
- Prompt: 圈出施工现场设备邻近作业人员的危险区域。
- Base IoU: `0.1739` | Tuned IoU: `0.4675` | Delta: `+0.2936`
- COCO source boxes: `1` | COCO target boxes: `1` | LISA polygons: `2`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0000_val__002__-equipment_proximity-114-_jpg.rf.07492924a04683042df5548fae45af02__equipment_proximity_coco_source.jpg) | ![](comparison_assets/0000_val__002__-equipment_proximity-114-_jpg.rf.07492924a04683042df5548fae45af02__equipment_proximity_coco_target.jpg) | ![](comparison_assets/0000_val__002__-equipment_proximity-114-_jpg.rf.07492924a04683042df5548fae45af02__equipment_proximity_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0000_val__002__-equipment_proximity-114-_jpg.rf.07492924a04683042df5548fae45af02__equipment_proximity_base_pred.jpg) | ![](comparison_assets/0000_val__002__-equipment_proximity-114-_jpg.rf.07492924a04683042df5548fae45af02__equipment_proximity_tuned_pred.jpg) |
