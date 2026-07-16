# Base / Tuned Sample Comparison

- Sample: `val__002__-harness_missing-199-_jpg.rf.613784a2a3a88e15b68ace04501a1544__harness_missing.jpg`
- Category: `harness_missing`
- Prompt: 把缺少安全带防护的作业人员分割出来。
- Base IoU: `0.3275` | Tuned IoU: `0.4512` | Delta: `+0.1237`
- COCO source boxes: `1` | COCO target boxes: `1` | LISA polygons: `5`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0007_val__002__-harness_missing-199-_jpg.rf.613784a2a3a88e15b68ace04501a1544__harness_missing_coco_source.jpg) | ![](comparison_assets/0007_val__002__-harness_missing-199-_jpg.rf.613784a2a3a88e15b68ace04501a1544__harness_missing_coco_target.jpg) | ![](comparison_assets/0007_val__002__-harness_missing-199-_jpg.rf.613784a2a3a88e15b68ace04501a1544__harness_missing_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0007_val__002__-harness_missing-199-_jpg.rf.613784a2a3a88e15b68ace04501a1544__harness_missing_base_pred.jpg) | ![](comparison_assets/0007_val__002__-harness_missing-199-_jpg.rf.613784a2a3a88e15b68ace04501a1544__harness_missing_tuned_pred.jpg) |
