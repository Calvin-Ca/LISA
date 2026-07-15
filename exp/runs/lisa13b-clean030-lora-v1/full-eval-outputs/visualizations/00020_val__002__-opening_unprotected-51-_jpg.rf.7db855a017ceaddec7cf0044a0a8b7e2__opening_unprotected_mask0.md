# Base / Tuned Sample Comparison

- Sample: `val__002__-opening_unprotected-51-_jpg.rf.7db855a017ceaddec7cf0044a0a8b7e2__opening_unprotected.jpg`
- Category: `opening_unprotected`
- Prompt: 图中哪些洞口或临边没有做防护?请分割出来。
- Base IoU: `0.0053` | Tuned IoU: `0.0000` | Delta: `-0.0053`
- COCO source boxes: `2` | COCO target boxes: `2` | LISA polygons: `2`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0020_val__002__-opening_unprotected-51-_jpg.rf.7db855a017ceaddec7cf0044a0a8b7e2__opening_unprotected_coco_source.jpg) | ![](comparison_assets/0020_val__002__-opening_unprotected-51-_jpg.rf.7db855a017ceaddec7cf0044a0a8b7e2__opening_unprotected_coco_target.jpg) | ![](comparison_assets/0020_val__002__-opening_unprotected-51-_jpg.rf.7db855a017ceaddec7cf0044a0a8b7e2__opening_unprotected_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0020_val__002__-opening_unprotected-51-_jpg.rf.7db855a017ceaddec7cf0044a0a8b7e2__opening_unprotected_base_pred.jpg) | ![](comparison_assets/0020_val__002__-opening_unprotected-51-_jpg.rf.7db855a017ceaddec7cf0044a0a8b7e2__opening_unprotected_tuned_pred.jpg) |
