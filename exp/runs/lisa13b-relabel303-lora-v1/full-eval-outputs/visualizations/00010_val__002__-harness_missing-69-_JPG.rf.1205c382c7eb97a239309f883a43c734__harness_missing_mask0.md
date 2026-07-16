# Base / Tuned Sample Comparison

- Sample: `val__002__-harness_missing-69-_JPG.rf.1205c382c7eb97a239309f883a43c734__harness_missing.jpg`
- Category: `harness_missing`
- Prompt: 把缺少安全带防护的作业人员分割出来。
- Base IoU: `0.7045` | Tuned IoU: `0.6183` | Delta: `-0.0861`
- COCO source boxes: `1` | COCO target boxes: `1` | LISA polygons: `8`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0010_val__002__-harness_missing-69-_JPG.rf.1205c382c7eb97a239309f883a43c734__harness_missing_coco_source.jpg) | ![](comparison_assets/0010_val__002__-harness_missing-69-_JPG.rf.1205c382c7eb97a239309f883a43c734__harness_missing_coco_target.jpg) | ![](comparison_assets/0010_val__002__-harness_missing-69-_JPG.rf.1205c382c7eb97a239309f883a43c734__harness_missing_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0010_val__002__-harness_missing-69-_JPG.rf.1205c382c7eb97a239309f883a43c734__harness_missing_base_pred.jpg) | ![](comparison_assets/0010_val__002__-harness_missing-69-_JPG.rf.1205c382c7eb97a239309f883a43c734__harness_missing_tuned_pred.jpg) |
