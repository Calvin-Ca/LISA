# Base / Tuned Sample Comparison

- Sample: `val__002__-opening_unprotected-14-_jpg.rf.f93c1e97c9f6ccaf8d87d079db200754__opening_unprotected.jpg`
- Category: `opening_unprotected`
- Prompt: 图中哪些洞口或临边没有做防护?请分割出来。
- Base IoU: `0.7369` | Tuned IoU: `0.8990` | Delta: `+0.1621`
- COCO source boxes: `1` | COCO target boxes: `1` | LISA polygons: `4`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0016_val__002__-opening_unprotected-14-_jpg.rf.f93c1e97c9f6ccaf8d87d079db200754__opening_unprotected_coco_source.jpg) | ![](comparison_assets/0016_val__002__-opening_unprotected-14-_jpg.rf.f93c1e97c9f6ccaf8d87d079db200754__opening_unprotected_coco_target.jpg) | ![](comparison_assets/0016_val__002__-opening_unprotected-14-_jpg.rf.f93c1e97c9f6ccaf8d87d079db200754__opening_unprotected_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0016_val__002__-opening_unprotected-14-_jpg.rf.f93c1e97c9f6ccaf8d87d079db200754__opening_unprotected_base_pred.jpg) | ![](comparison_assets/0016_val__002__-opening_unprotected-14-_jpg.rf.f93c1e97c9f6ccaf8d87d079db200754__opening_unprotected_tuned_pred.jpg) |
