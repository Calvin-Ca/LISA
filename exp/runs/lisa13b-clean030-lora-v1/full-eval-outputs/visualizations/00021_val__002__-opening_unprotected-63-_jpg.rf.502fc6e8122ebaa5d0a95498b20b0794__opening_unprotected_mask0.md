# Base / Tuned Sample Comparison

- Sample: `val__002__-opening_unprotected-63-_jpg.rf.502fc6e8122ebaa5d0a95498b20b0794__opening_unprotected.jpg`
- Category: `opening_unprotected`
- Prompt: 图中哪些洞口或临边没有做防护?请分割出来。
- Base IoU: `0.1233` | Tuned IoU: `0.1997` | Delta: `+0.0764`
- COCO source boxes: `1` | COCO target boxes: `1` | LISA polygons: `20`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0021_val__002__-opening_unprotected-63-_jpg.rf.502fc6e8122ebaa5d0a95498b20b0794__opening_unprotected_coco_source.jpg) | ![](comparison_assets/0021_val__002__-opening_unprotected-63-_jpg.rf.502fc6e8122ebaa5d0a95498b20b0794__opening_unprotected_coco_target.jpg) | ![](comparison_assets/0021_val__002__-opening_unprotected-63-_jpg.rf.502fc6e8122ebaa5d0a95498b20b0794__opening_unprotected_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0021_val__002__-opening_unprotected-63-_jpg.rf.502fc6e8122ebaa5d0a95498b20b0794__opening_unprotected_base_pred.jpg) | ![](comparison_assets/0021_val__002__-opening_unprotected-63-_jpg.rf.502fc6e8122ebaa5d0a95498b20b0794__opening_unprotected_tuned_pred.jpg) |
