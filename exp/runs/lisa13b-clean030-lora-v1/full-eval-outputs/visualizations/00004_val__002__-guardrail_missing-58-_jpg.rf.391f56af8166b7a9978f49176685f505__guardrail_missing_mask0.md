# Base / Tuned Sample Comparison

- Sample: `val__002__-guardrail_missing-58-_jpg.rf.391f56af8166b7a9978f49176685f505__guardrail_missing.jpg`
- Category: `guardrail_missing`
- Prompt: 把缺少栏杆防护、存在坠落风险的部位分割出来。
- Base IoU: `0.0000` | Tuned IoU: `0.1178` | Delta: `+0.1178`
- COCO source boxes: `1` | COCO target boxes: `1` | LISA polygons: `11`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0004_val__002__-guardrail_missing-58-_jpg.rf.391f56af8166b7a9978f49176685f505__guardrail_missing_coco_source.jpg) | ![](comparison_assets/0004_val__002__-guardrail_missing-58-_jpg.rf.391f56af8166b7a9978f49176685f505__guardrail_missing_coco_target.jpg) | ![](comparison_assets/0004_val__002__-guardrail_missing-58-_jpg.rf.391f56af8166b7a9978f49176685f505__guardrail_missing_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0004_val__002__-guardrail_missing-58-_jpg.rf.391f56af8166b7a9978f49176685f505__guardrail_missing_base_pred.jpg) | ![](comparison_assets/0004_val__002__-guardrail_missing-58-_jpg.rf.391f56af8166b7a9978f49176685f505__guardrail_missing_tuned_pred.jpg) |
