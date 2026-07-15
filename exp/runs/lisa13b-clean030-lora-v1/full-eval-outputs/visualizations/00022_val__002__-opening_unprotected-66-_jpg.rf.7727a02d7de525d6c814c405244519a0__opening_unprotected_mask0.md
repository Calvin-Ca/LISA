# Base / Tuned Sample Comparison

- Sample: `val__002__-opening_unprotected-66-_jpg.rf.7727a02d7de525d6c814c405244519a0__opening_unprotected.jpg`
- Category: `opening_unprotected`
- Prompt: 图中哪些洞口或临边没有做防护?请分割出来。
- Base IoU: `0.0627` | Tuned IoU: `0.5765` | Delta: `+0.5138`
- COCO source boxes: `1` | COCO target boxes: `1` | LISA polygons: `3`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0022_val__002__-opening_unprotected-66-_jpg.rf.7727a02d7de525d6c814c405244519a0__opening_unprotected_coco_source.jpg) | ![](comparison_assets/0022_val__002__-opening_unprotected-66-_jpg.rf.7727a02d7de525d6c814c405244519a0__opening_unprotected_coco_target.jpg) | ![](comparison_assets/0022_val__002__-opening_unprotected-66-_jpg.rf.7727a02d7de525d6c814c405244519a0__opening_unprotected_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0022_val__002__-opening_unprotected-66-_jpg.rf.7727a02d7de525d6c814c405244519a0__opening_unprotected_base_pred.jpg) | ![](comparison_assets/0022_val__002__-opening_unprotected-66-_jpg.rf.7727a02d7de525d6c814c405244519a0__opening_unprotected_tuned_pred.jpg) |
