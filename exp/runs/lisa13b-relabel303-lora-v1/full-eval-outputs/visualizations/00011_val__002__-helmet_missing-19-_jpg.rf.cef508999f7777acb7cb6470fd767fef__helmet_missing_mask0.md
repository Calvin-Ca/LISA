# Base / Tuned Sample Comparison

- Sample: `val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing.jpg`
- Category: `helmet_missing`
- Prompt: 标出未按规定佩戴安全帽的作业人员。
- Base IoU: `0.5571` | Tuned IoU: `0.9185` | Delta: `+0.3614`
- COCO source boxes: `1` | COCO target boxes: `1` | LISA polygons: `1`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0011_val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing_coco_source.jpg) | ![](comparison_assets/0011_val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing_coco_target.jpg) | ![](comparison_assets/0011_val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0011_val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing_base_pred.jpg) | ![](comparison_assets/0011_val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing_tuned_pred.jpg) |
