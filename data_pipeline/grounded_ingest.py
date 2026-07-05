"""
Stage 0- · 无标注数据前端(grounded ingest)

用途:当原料是"没有包围框的原始照片 / 视频"时,自动补上框,
让它们接得上 build_dataset.py。整条链零人工标注:

    原始图片/视频
      -> (视频)抽帧 + 去重
      -> Grounding DINO 开放词表检测   得到"基础实体"框(person / helmet / vest / wire ...)
      -> 几何规则推导隐患             (如:person 头部区域无 helmet 重叠 => 未戴安全帽)
      -> 输出 <name>.jpg + <name>.txt (class x1 y1 x2 y2)  <== build_dataset.py 的输入格式

关键设计点(面试可讲):
  - 开放词表检测器擅长"实体",不擅长"否定/组合语义"(如 "person without helmet")。
    所以先检测基础实体,再用几何规则组合出隐患 —— 把"组合语义"留给规则,而非硬塞给检测器。
  - 全自动出框有噪声,务必保留 build_dataset 里的面积/IoU 过滤 + 人工抽检。

依赖:
  - 抽帧只需 opencv,可独立运行(--frames-only)。
  - 检测需 GroundingDINO(惰性导入),未安装时给出清晰提示。
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from config import RAW_DIR

# ------------------------------------------------------------------ 配置
INGEST_DIR = Path(__file__).resolve().parent / "ingest"   # 放原始图片/视频

# 送给开放词表检测器的"基础实体"文字提示(注意:是实体,不是隐患)
BASE_ENTITY_PROMPT = "person . head . helmet . reflective vest . electrical wire"

DET = {
    "box_thresh": 0.30,       # 检测框置信度阈值
    "text_thresh": 0.25,
    "head_region_ratio": 0.35,  # person 框顶部多少比例视为"头部区域"
    "torso_top": 0.25,          # person 框躯干区域(占比)上边界
    "torso_bottom": 0.65,       # 躯干区域下边界
    "overlap_thresh": 0.10,     # 实体重叠判定阈值(IoU / 包含比)
}

FRAME = {
    "every_n": 15,            # 每隔 N 帧取一帧(粗采样)
    "scene_diff": 0.35,       # 与上一保留帧的差异高于此值才保留(去重)
}


# ------------------------------------------------------------------ 几何工具
def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-6)


def _overlap_in_region(entity_box, region):
    """entity_box 落在 region 内的面积 / region 面积(判断头/躯干区域里有没有该实体)。"""
    ix1, iy1 = max(entity_box[0], region[0]), max(entity_box[1], region[1])
    ix2, iy2 = min(entity_box[2], region[2]), min(entity_box[3], region[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ra = (region[2] - region[0]) * (region[3] - region[1])
    return inter / (ra + 1e-6)


# ------------------------------------------------------------------ Stage: 隐患规则推导
def derive_hazards(dets):
    """
    dets: [{"label": str, "box": [x1,y1,x2,y2]}]  基础实体检测结果
    返回: [(hazard_class, [x1,y1,x2,y2])]          隐患框(class 需在 config.CLASS_TO_HAZARD 里)
    """
    persons = [d["box"] for d in dets if d["label"] == "person"]
    helmets = [d["box"] for d in dets if d["label"] == "helmet"]
    vests = [d["box"] for d in dets if d["label"] in ("reflective vest", "vest")]
    wires = [d["box"] for d in dets if d["label"] in ("electrical wire", "wire")]

    out = []
    for p in persons:
        x1, y1, x2, y2 = p
        h = y2 - y1
        # 头部区域 = person 框顶部一段
        head_region = [x1, y1, x2, y1 + h * DET["head_region_ratio"]]
        # 躯干区域
        torso_region = [x1, y1 + h * DET["torso_top"], x2, y1 + h * DET["torso_bottom"]]

        # 规则1:头部区域没有任何 helmet 重叠 => 未戴安全帽
        if not any(_overlap_in_region(hm, head_region) > DET["overlap_thresh"] for hm in helmets):
            out.append(("person_no_helmet", p))

        # 规则2:躯干区域没有任何 vest 重叠 => 未穿反光衣
        if not any(_overlap_in_region(v, torso_region) > DET["overlap_thresh"] for v in vests):
            out.append(("person_no_vest", p))

    # 规则3:裸露电线是"直接实体",检测到即隐患(非组合语义)
    for w in wires:
        out.append(("exposed_wire", w))

    # 注:edge_no_guardrail(临边无防护)需场景级理解,规则难覆盖,留待人工/专用模型补充
    return out


# ------------------------------------------------------------------ Stage: 视频抽帧 + 去重
def extract_frames(video_path, out_dir, every_n, scene_diff):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    stem = video_path.stem
    idx, kept, last_hist = 0, 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
            hist = cv2.normalize(hist, hist).flatten()
            # 与上一保留帧差异够大才保留(去重相似帧)
            if last_hist is None or (1 - cv2.compareHist(last_hist, hist, cv2.HISTCMP_CORREL)) > scene_diff:
                cv2.imwrite(str(out_dir / f"{stem}_f{idx:06d}.jpg"), frame)
                last_hist = hist
                kept += 1
        idx += 1
    cap.release()
    print(f"  [抽帧] {video_path.name}: {idx} 帧 -> 保留 {kept} 帧")
    return kept


# ------------------------------------------------------------------ 检测器(惰性导入)
class GroundingDetector:
    """封装 Grounding DINO;未安装时给出安装提示,不影响 --frames-only。"""

    def __init__(self, prompt=BASE_ENTITY_PROMPT):
        self.prompt = prompt
        try:
            from groundingdino.util.inference import Model  # noqa
        except ImportError as e:
            raise ImportError(
                "未安装 GroundingDINO。请:\n"
                "  pip install groundingdino-py\n"
                "  下载权重 groundingdino_swint_ogc.pth 与配置,\n"
                "然后在此处补全 Model 初始化。"
            ) from e
        # TODO: 用你的权重/配置初始化
        #   self.model = Model(model_config_path=..., model_checkpoint_path=...)
        raise NotImplementedError("补全 GroundingDINO Model 初始化后即可使用")

    def detect(self, image_bgr) -> list:
        """返回 [{"label": str, "box": [x1,y1,x2,y2]}](像素坐标)。"""
        raise NotImplementedError


# ------------------------------------------------------------------ 主流程
def ingest(frames_only=False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frame_dir = INGEST_DIR / "_frames"

    # 1) 视频 -> 帧
    videos = [p for p in INGEST_DIR.glob("*") if p.suffix.lower() in (".mp4", ".mov", ".avi")]
    for v in videos:
        extract_frames(v, frame_dir, FRAME["every_n"], FRAME["scene_diff"])

    # 收集所有待处理图片(原始图 + 抽出的帧)
    images = [p for p in INGEST_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    images += list(frame_dir.glob("*.jpg"))
    print(f"[Ingest] 待处理图片 {len(images)} 张")

    if frames_only:
        print("[Ingest] --frames-only:仅抽帧,跳过检测。")
        return

    # 2) 检测 + 3) 规则推导 + 写出框
    detector = GroundingDetector()
    n_out = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        dets = detector.detect(img)
        hazards = derive_hazards(dets)
        if not hazards:
            continue
        stem = img_path.stem
        cv2.imwrite(str(RAW_DIR / f"{stem}.jpg"), img)
        lines = [f"{cls} {int(b[0])} {int(b[1])} {int(b[2])} {int(b[3])}" for cls, b in hazards]
        (RAW_DIR / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_out += 1

    print(f"[Ingest] 生成 {n_out} 个带框样本 -> {RAW_DIR}(接着跑 build_dataset.py)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-only", action="store_true",
                    help="只对视频抽帧(仅需 opencv,不加载检测器)")
    args = ap.parse_args()
    ingest(frames_only=args.frames_only)
