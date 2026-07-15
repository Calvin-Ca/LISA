# Base / Tuned Sample Comparison

- Sample: `val__002__-opening_unprotected-11-_jpg.rf.b006631ca767138c74d5c79ee727de86__opening_unprotected.jpg`
- Category: `opening_unprotected`
- Prompt: 圈出没有防护的洞口或临边区域。
- Base IoU: `0.1396` | Tuned IoU: `0.4617` | Delta: `+0.3221`
- COCO source boxes: `1` | COCO target boxes: `1` | LISA polygons: `46`

| COCO source annotations | COCO target annotations | LISA annotations |
| --- | --- | --- |
| ![](comparison_assets/0015_val__002__-opening_unprotected-11-_jpg.rf.b006631ca767138c74d5c79ee727de86__opening_unprotected_coco_source.jpg) | ![](comparison_assets/0015_val__002__-opening_unprotected-11-_jpg.rf.b006631ca767138c74d5c79ee727de86__opening_unprotected_coco_target.jpg) | ![](comparison_assets/0015_val__002__-opening_unprotected-11-_jpg.rf.b006631ca767138c74d5c79ee727de86__opening_unprotected_lisa.jpg) |

| Base benchmark prediction | Tuned prediction |
| --- | --- |
| ![](comparison_assets/0015_val__002__-opening_unprotected-11-_jpg.rf.b006631ca767138c74d5c79ee727de86__opening_unprotected_base_pred.jpg) | ![](comparison_assets/0015_val__002__-opening_unprotected-11-_jpg.rf.b006631ca767138c74d5c79ee727de86__opening_unprotected_tuned_pred.jpg) |
