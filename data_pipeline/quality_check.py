"""
Stage 6 · 质检可视化

把合成好的 <name>.jpg + <name>.json 叠加掩码画出来,供人工抽检。
复用与 utils/data_processing.py 相同的读取逻辑,确保"所见即训练所得"。
绿色=目标区域。图上打印对应的推理指令。
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.data_processing import get_mask_from_json  # noqa: E402

from config import OUT_DIR, VIS_DIR


def visualize(out_dir=OUT_DIR, vis_dir=VIS_DIR, limit=None):
    vis_dir.mkdir(parents=True, exist_ok=True)
    json_paths = sorted(out_dir.glob("*.json"))
    json_paths = [p for p in json_paths if p.name != "split.json"]
    if limit:
        json_paths = json_paths[:limit]

    for jp in json_paths:
        img_path = jp.with_suffix(".jpg")
        img = cv2.imread(str(img_path))[:, :, ::-1]
        mask, comments, is_sentence = get_mask_from_json(str(jp), img)

        valid = (mask == 1).astype(np.float32)[:, :, None]
        vis = img * (1 - valid) + (np.array([0, 255, 0]) * 0.5 + img * 0.5) * valid
        vis = np.concatenate([img, vis], axis=1).astype(np.uint8)

        text = comments[0] if comments else ""
        cv2.putText(vis[:, :, ::-1], text[:40], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(str(vis_dir / f"{jp.stem}.jpg"), vis[:, :, ::-1])

    print(f"[质检] 可视化 {len(json_paths)} 条 -> {vis_dir}")


if __name__ == "__main__":
    visualize()
