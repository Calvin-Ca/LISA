# Base / Tuned Sample Comparison

- Sample: `val__002__-guardrail_missing-64-_jpg.rf.ce53035dccc9877ee6e3ffad7fe23cd1__guardrail_missing.jpg`
- Category: `guardrail_missing`
- Prompt: 图中哪些位置存在防护栏杆缺失隐患?请分割出来。
- Base IoU: `0.1640` | Tuned IoU: `0.3472` | Delta: `+0.1832`
- COCO source boxes: `2` | COCO target boxes: `2` | LISA polygons: `10`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0006_val__002__-guardrail_missing-64-_jpg.rf.ce53035dccc9877ee6e3ffad7fe23cd1__guardrail_missing_coco_source.jpg) | ![](comparison_assets/0006_val__002__-guardrail_missing-64-_jpg.rf.ce53035dccc9877ee6e3ffad7fe23cd1__guardrail_missing_coco_target.jpg) | ![](comparison_assets/0006_val__002__-guardrail_missing-64-_jpg.rf.ce53035dccc9877ee6e3ffad7fe23cd1__guardrail_missing_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0006_val__002__-guardrail_missing-64-_jpg.rf.ce53035dccc9877ee6e3ffad7fe23cd1__guardrail_missing_base_pred.jpg) | ![](comparison_assets/0006_val__002__-guardrail_missing-64-_jpg.rf.ce53035dccc9877ee6e3ffad7fe23cd1__guardrail_missing_tuned_pred.jpg) |
