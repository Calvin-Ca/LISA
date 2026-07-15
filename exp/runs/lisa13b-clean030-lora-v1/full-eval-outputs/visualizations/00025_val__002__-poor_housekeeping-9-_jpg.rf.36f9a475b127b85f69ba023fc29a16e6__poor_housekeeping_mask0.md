# Base / Tuned Sample Comparison

- Sample: `val__002__-poor_housekeeping-9-_jpg.rf.36f9a475b127b85f69ba023fc29a16e6__poor_housekeeping.jpg`
- Category: `poor_housekeeping`
- Prompt: 标出现场材料堆放混乱或文明施工不到位的区域。
- Base IoU: `0.3122` | Tuned IoU: `0.8270` | Delta: `+0.5149`
- COCO source boxes: `1` | COCO target boxes: `1` | LISA polygons: `3`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0025_val__002__-poor_housekeeping-9-_jpg.rf.36f9a475b127b85f69ba023fc29a16e6__poor_housekeeping_coco_source.jpg) | ![](comparison_assets/0025_val__002__-poor_housekeeping-9-_jpg.rf.36f9a475b127b85f69ba023fc29a16e6__poor_housekeeping_coco_target.jpg) | ![](comparison_assets/0025_val__002__-poor_housekeeping-9-_jpg.rf.36f9a475b127b85f69ba023fc29a16e6__poor_housekeeping_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0025_val__002__-poor_housekeeping-9-_jpg.rf.36f9a475b127b85f69ba023fc29a16e6__poor_housekeeping_base_pred.jpg) | ![](comparison_assets/0025_val__002__-poor_housekeeping-9-_jpg.rf.36f9a475b127b85f69ba023fc29a16e6__poor_housekeeping_tuned_pred.jpg) |
