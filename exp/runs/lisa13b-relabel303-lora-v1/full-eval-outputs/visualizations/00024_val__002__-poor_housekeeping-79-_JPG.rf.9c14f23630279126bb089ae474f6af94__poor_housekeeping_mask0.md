# Base / Tuned Sample Comparison

- Sample: `val__002__-poor_housekeeping-79-_JPG.rf.9c14f23630279126bb089ae474f6af94__poor_housekeeping.jpg`
- Category: `poor_housekeeping`
- Prompt: 圈出施工现场杂乱、可能影响通行安全的位置。
- Base IoU: `0.0438` | Tuned IoU: `0.5549` | Delta: `+0.5112`
- COCO source boxes: `3` | COCO target boxes: `3` | LISA polygons: `41`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0024_val__002__-poor_housekeeping-79-_JPG.rf.9c14f23630279126bb089ae474f6af94__poor_housekeeping_coco_source.jpg) | ![](comparison_assets/0024_val__002__-poor_housekeeping-79-_JPG.rf.9c14f23630279126bb089ae474f6af94__poor_housekeeping_coco_target.jpg) | ![](comparison_assets/0024_val__002__-poor_housekeeping-79-_JPG.rf.9c14f23630279126bb089ae474f6af94__poor_housekeeping_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0024_val__002__-poor_housekeeping-79-_JPG.rf.9c14f23630279126bb089ae474f6af94__poor_housekeeping_base_pred.jpg) | ![](comparison_assets/0024_val__002__-poor_housekeeping-79-_JPG.rf.9c14f23630279126bb089ae474f6af94__poor_housekeeping_tuned_pred.jpg) |
