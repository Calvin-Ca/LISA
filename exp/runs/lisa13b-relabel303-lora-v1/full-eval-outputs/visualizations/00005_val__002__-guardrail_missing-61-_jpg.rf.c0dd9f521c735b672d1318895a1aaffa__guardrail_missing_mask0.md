# Base / Tuned Sample Comparison

- Sample: `val__002__-guardrail_missing-61-_jpg.rf.c0dd9f521c735b672d1318895a1aaffa__guardrail_missing.jpg`
- Category: `guardrail_missing`
- Prompt: 指出没有设置防护栏杆的临边区域。
- Base IoU: `0.0000` | Tuned IoU: `0.0093` | Delta: `+0.0093`
- COCO source boxes: `1` | COCO target boxes: `1` | LISA polygons: `6`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0005_val__002__-guardrail_missing-61-_jpg.rf.c0dd9f521c735b672d1318895a1aaffa__guardrail_missing_coco_source.jpg) | ![](comparison_assets/0005_val__002__-guardrail_missing-61-_jpg.rf.c0dd9f521c735b672d1318895a1aaffa__guardrail_missing_coco_target.jpg) | ![](comparison_assets/0005_val__002__-guardrail_missing-61-_jpg.rf.c0dd9f521c735b672d1318895a1aaffa__guardrail_missing_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0005_val__002__-guardrail_missing-61-_jpg.rf.c0dd9f521c735b672d1318895a1aaffa__guardrail_missing_base_pred.jpg) | ![](comparison_assets/0005_val__002__-guardrail_missing-61-_jpg.rf.c0dd9f521c735b672d1318895a1aaffa__guardrail_missing_tuned_pred.jpg) |
