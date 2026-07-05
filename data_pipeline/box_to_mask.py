"""
Stage 1 · 框 → 像素掩码

复用本仓库自带的 SAM(model/segment_anything),用检测框作为 box prompt,
把粗糙的 bbox 提升为精细的像素级掩码 —— 这是把"检测数据"升级成
"分割数据"的关键一步,且零人工标注成本。
"""

import sys
from pathlib import Path

import numpy as np
import torch

# 让脚本能 import 到仓库根目录下的 model 包
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.segment_anything import SamPredictor, sam_model_registry  # noqa: E402


class BoxToMask:
    def __init__(self, checkpoint, model_type="vit_h", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(self.device)
        self.predictor = SamPredictor(sam)

    def set_image(self, image_rgb: np.ndarray):
        """image_rgb: HxWx3, RGB, uint8。每张图设一次,可复用于多个框。"""
        self.predictor.set_image(image_rgb)

    def boxes_to_masks(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        """
        boxes_xyxy: [N,4]，返回 [N,H,W] 的 0/1 掩码。
        对每个框取 SAM 置信度最高的一张掩码。
        """
        if len(boxes_xyxy) == 0:
            return np.zeros((0, *self.predictor.original_size), dtype=np.uint8)

        boxes = torch.as_tensor(boxes_xyxy, dtype=torch.float, device=self.device)
        transformed = self.predictor.transform.apply_boxes_torch(
            boxes, self.predictor.original_size
        )
        masks, scores, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=False,   # 每框一张最优掩码
        )
        return masks[:, 0].cpu().numpy().astype(np.uint8)  # [N,H,W]


def box_mask_iou(box_xyxy, mask):
    """SAM 掩码与原始框的 IoU,用于质检(掩码是否跑偏)。"""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    if box_area == 0:
        return 0.0
    inter = mask[y1:y2, x1:x2].sum()
    union = mask.sum() + box_area - inter
    return float(inter) / float(union + 1e-6)
