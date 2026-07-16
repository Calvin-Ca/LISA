# Base / Tuned Sample Comparison

- Sample: `val__002__-helmet_missing-249-_JPG.rf.161f677eff339180405f881cbd790a96__helmet_missing.jpg`
- Category: `helmet_missing`
- Prompt: 现场哪些人员存在未戴安全帽的安全隐患?请分割出来。
- Base IoU: `0.0144` | Tuned IoU: `0.0133` | Delta: `-0.0010`
- COCO source boxes: `1` | COCO target boxes: `1` | LISA polygons: `3`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0014_val__002__-helmet_missing-249-_JPG.rf.161f677eff339180405f881cbd790a96__helmet_missing_coco_source.jpg) | ![](comparison_assets/0014_val__002__-helmet_missing-249-_JPG.rf.161f677eff339180405f881cbd790a96__helmet_missing_coco_target.jpg) | ![](comparison_assets/0014_val__002__-helmet_missing-249-_JPG.rf.161f677eff339180405f881cbd790a96__helmet_missing_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0014_val__002__-helmet_missing-249-_JPG.rf.161f677eff339180405f881cbd790a96__helmet_missing_base_pred.jpg) | ![](comparison_assets/0014_val__002__-helmet_missing-249-_JPG.rf.161f677eff339180405f881cbd790a96__helmet_missing_tuned_pred.jpg) |
