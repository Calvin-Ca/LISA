# Base / Tuned Sample Comparison

- Sample: `val__002__-poor_housekeeping-93-_jpg.rf.893e8f8298279e8c46d0fa8a503d58ff__poor_housekeeping.jpg`
- Category: `poor_housekeeping`
- Prompt: 圈出施工现场杂乱、可能影响通行安全的位置。
- Base IoU: `0.0000` | Tuned IoU: `0.9596` | Delta: `+0.9596`
- COCO source boxes: `1` | COCO target boxes: `1` | LISA polygons: `1`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0026_val__002__-poor_housekeeping-93-_jpg.rf.893e8f8298279e8c46d0fa8a503d58ff__poor_housekeeping_coco_source.jpg) | ![](comparison_assets/0026_val__002__-poor_housekeeping-93-_jpg.rf.893e8f8298279e8c46d0fa8a503d58ff__poor_housekeeping_coco_target.jpg) | ![](comparison_assets/0026_val__002__-poor_housekeeping-93-_jpg.rf.893e8f8298279e8c46d0fa8a503d58ff__poor_housekeeping_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0026_val__002__-poor_housekeeping-93-_jpg.rf.893e8f8298279e8c46d0fa8a503d58ff__poor_housekeeping_base_pred.jpg) | ![](comparison_assets/0026_val__002__-poor_housekeeping-93-_jpg.rf.893e8f8298279e8c46d0fa8a503d58ff__poor_housekeeping_tuned_pred.jpg) |
