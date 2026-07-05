"""
数据合成 pipeline 配置(场景 A · 施工安全隐患巡检)

核心思想:把现成"检测框"数据集,自动转成 LISA 需要的"推理指令 → 像素掩码"数据。
本文件定义:隐患分类体系(taxonomy)+ 检测类别到隐患的映射 + 输出规格。
"""

from pathlib import Path

# ------------------------------------------------------------------ 路径
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"            # 原料:图片 + 检测标注(bbox)
OUT_DIR = ROOT / "out"            # 产物:LISA 格式 jpg + json
VIS_DIR = ROOT / "vis"            # 质检可视化
SAM_CHECKPOINT = ROOT / "sam_vit_h_4b8939.pth"   # SAM 权重(框→掩码用)
SAM_MODEL_TYPE = "vit_h"

# ------------------------------------------------------------------ 隐患分类体系
# 每个隐患项定义:
#   source_classes: 该隐患对应原始检测数据集里的哪些类别
#   is_sentence:    True=推理句式(LISA 长指令),False=短语类别式
#   severity:       严重等级(写进工单;也可做采样均衡)
#   regulation:     关联规范条文(供 Agent 端 RAG,数据阶段仅记录)
HAZARD_TAXONOMY = {
    "no_helmet": {
        "source_classes": ["head", "person_no_helmet", "no-helmet", "nohardhat"],
        "is_sentence": True,
        "severity": "high",
        "regulation": "JGJ59-2011 3.13 施工现场人员未按规定佩戴安全帽",
    },
    "no_reflective_vest": {
        "source_classes": ["person_no_vest", "no-vest"],
        "is_sentence": True,
        "severity": "medium",
        "regulation": "施工现场作业人员未穿戴反光衣",
    },
    "edge_no_guardrail": {
        # 临边无防护:多为组合语义,现成数据少,常需人工/规则补充
        "source_classes": ["open_edge", "unprotected_edge"],
        "is_sentence": True,
        "severity": "high",
        "regulation": "JGJ59-2011 临边洞口未设置防护栏杆",
    },
    "exposed_wiring": {
        "source_classes": ["exposed_wire", "loose_cable"],
        "is_sentence": True,
        "severity": "high",
        "regulation": "临时用电电线裸露/敷设不规范",
    },
}

# ------------------------------------------------------------------ 质检阈值
QC = {
    "min_mask_area_ratio": 0.0008,   # 掩码面积 / 图面积 下限,过滤碎块误分
    "max_mask_area_ratio": 0.6,      # 上限,过滤把整图当目标
    "min_box_mask_iou": 0.3,         # SAM 掩码与原框的 IoU 下限,过滤跑偏
    "poly_epsilon_ratio": 0.005,     # 多边形简化程度(占周长比例)
    "poly_min_points": 3,
}

# ------------------------------------------------------------------ 划分
VAL_RATIO = 0.1
RANDOM_SEED = 42
