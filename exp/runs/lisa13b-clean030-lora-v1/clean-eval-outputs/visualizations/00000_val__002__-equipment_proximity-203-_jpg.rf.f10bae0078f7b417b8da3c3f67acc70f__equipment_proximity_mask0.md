# Base / Tuned Sample Comparison

- Sample: `val__002__-equipment_proximity-203-_jpg.rf.f10bae0078f7b417b8da3c3f67acc70f__equipment_proximity.jpg`
- Category: `equipment_proximity`
- Prompt: 图中哪些位置存在设备靠近人员的安全隐患?请分割出来。
- Base IoU: `0.6013` | Tuned IoU: `0.7441` | Delta: `+0.1429`
- COCO source boxes: `1` | COCO target boxes: `1` | LISA polygons: `27`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0000_val__002__-equipment_proximity-203-_jpg.rf.f10bae0078f7b417b8da3c3f67acc70f__equipment_proximity_coco_source.jpg) | ![](comparison_assets/0000_val__002__-equipment_proximity-203-_jpg.rf.f10bae0078f7b417b8da3c3f67acc70f__equipment_proximity_coco_target.jpg) | ![](comparison_assets/0000_val__002__-equipment_proximity-203-_jpg.rf.f10bae0078f7b417b8da3c3f67acc70f__equipment_proximity_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0000_val__002__-equipment_proximity-203-_jpg.rf.f10bae0078f7b417b8da3c3f67acc70f__equipment_proximity_base_pred.jpg) | ![](comparison_assets/0000_val__002__-equipment_proximity-203-_jpg.rf.f10bae0078f7b417b8da3c3f67acc70f__equipment_proximity_tuned_pred.jpg) |
