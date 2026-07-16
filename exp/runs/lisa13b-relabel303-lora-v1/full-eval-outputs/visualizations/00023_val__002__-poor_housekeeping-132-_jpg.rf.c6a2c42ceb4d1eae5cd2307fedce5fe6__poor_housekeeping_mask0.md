# Base / Tuned Sample Comparison

- Sample: `val__002__-poor_housekeeping-132-_jpg.rf.c6a2c42ceb4d1eae5cd2307fedce5fe6__poor_housekeeping.jpg`
- Category: `poor_housekeeping`
- Prompt: 图中哪些区域存在场地整理不到位的安全隐患?请分割出来。
- Base IoU: `0.5302` | Tuned IoU: `0.0000` | Delta: `-0.5302`
- COCO source boxes: `4` | COCO target boxes: `4` | LISA polygons: `5`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0023_val__002__-poor_housekeeping-132-_jpg.rf.c6a2c42ceb4d1eae5cd2307fedce5fe6__poor_housekeeping_coco_source.jpg) | ![](comparison_assets/0023_val__002__-poor_housekeeping-132-_jpg.rf.c6a2c42ceb4d1eae5cd2307fedce5fe6__poor_housekeeping_coco_target.jpg) | ![](comparison_assets/0023_val__002__-poor_housekeeping-132-_jpg.rf.c6a2c42ceb4d1eae5cd2307fedce5fe6__poor_housekeeping_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0023_val__002__-poor_housekeeping-132-_jpg.rf.c6a2c42ceb4d1eae5cd2307fedce5fe6__poor_housekeeping_base_pred.jpg) | ![](comparison_assets/0023_val__002__-poor_housekeeping-132-_jpg.rf.c6a2c42ceb4d1eae5cd2307fedce5fe6__poor_housekeeping_tuned_pred.jpg) |
