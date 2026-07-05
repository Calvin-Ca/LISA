"""
把合成产物(out/)按 split.json 分发到 LISA 训练目录:
    dataset/reason_seg/ReasonSeg/train/
    dataset/reason_seg/ReasonSeg/val/

用法:
    python deploy_to_dataset.py
    python deploy_to_dataset.py --move   # 移动而非复制(省磁盘)
"""

import argparse
import json
import shutil
from pathlib import Path

from config import OUT_DIR

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "reason_seg" / "ReasonSeg"


def deploy(move=False):
    split_file = OUT_DIR / "split.json"
    if not split_file.exists():
        raise FileNotFoundError(f"未找到 {split_file},请先运行 build_dataset.py")

    split = json.loads(split_file.read_text(encoding="utf-8"))
    op = shutil.move if move else shutil.copy2

    n = {"train": 0, "val": 0}
    for part in ("train", "val"):
        dst_dir = DATASET_ROOT / part
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in split.get(part, []):
            for ext in (".jpg", ".json"):
                src = OUT_DIR / f"{name}{ext}"
                if src.exists():
                    op(str(src), str(dst_dir / f"{name}{ext}"))
            n[part] += 1

    print(f"[部署] train={n['train']}  val={n['val']}  -> {DATASET_ROOT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--move", action="store_true", help="移动而非复制")
    args = ap.parse_args()
    deploy(move=args.move)
