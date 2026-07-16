# Base / Tuned Sample Comparison

- Sample: `val__002__-harness_missing-23-_JPG.rf.9e5a10303ac02c86c71739d108101ea4__harness_missing.jpg`
- Category: `harness_missing`
- Prompt: 把缺少安全带防护的作业人员分割出来。
- Base IoU: `0.4646` | Tuned IoU: `0.4652` | Delta: `+0.0005`
- COCO source boxes: `2` | COCO target boxes: `2` | LISA polygons: `10`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0008_val__002__-harness_missing-23-_JPG.rf.9e5a10303ac02c86c71739d108101ea4__harness_missing_coco_source.jpg) | ![](comparison_assets/0008_val__002__-harness_missing-23-_JPG.rf.9e5a10303ac02c86c71739d108101ea4__harness_missing_coco_target.jpg) | ![](comparison_assets/0008_val__002__-harness_missing-23-_JPG.rf.9e5a10303ac02c86c71739d108101ea4__harness_missing_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0008_val__002__-harness_missing-23-_JPG.rf.9e5a10303ac02c86c71739d108101ea4__harness_missing_base_pred.jpg) | ![](comparison_assets/0008_val__002__-harness_missing-23-_JPG.rf.9e5a10303ac02c86c71739d108101ea4__harness_missing_tuned_pred.jpg) |
