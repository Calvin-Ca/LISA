"""
Stage 3 · 推理指令生成

把"隐患类型"转成自然语言的推理式指令(reasoning instruction)。
两种来源:
  1) 模板库(template bank)—— 稳定、可控、零成本;
  2) LLM 改写(可选)—— 增加表达多样性,避免模型过拟合到固定句式。

设计要点:LISA 的价值在"推理分割",所以指令要尽量写成
组合语义 / 意图描述,而不是简单类别名。
"""

import random

# 每个隐患类型的模板库(推理句式,is_sentence=True 时使用)
TEMPLATE_BANK = {
    "no_helmet": [
        "圈出图中没有佩戴安全帽的工人。",
        "标出未按规定佩戴安全帽的作业人员。",
        "现场哪些人员存在未戴安全帽的安全隐患?请分割出来。",
        "把没有做好头部防护、未戴安全帽的人框出来。",
    ],
    "no_reflective_vest": [
        "圈出未穿反光衣的作业人员。",
        "标出没有按要求穿戴反光背心的工人。",
        "现场哪些人没有穿反光衣?请分割。",
    ],
    "edge_no_guardrail": [
        "指出没有设置防护栏杆的临边区域。",
        "标出存在临边坠落风险、缺少防护的部位。",
        "图中哪些临边或洞口没有做防护?请分割出来。",
    ],
    "exposed_wiring": [
        "找出裸露的电线或敷设不规范的线缆。",
        "标出存在漏电风险的裸露电线区域。",
        "图中哪里的临时用电存在电线裸露隐患?请分割。",
    ],
}

# 短语式(is_sentence=False 时使用,作为少量补充,增强类别泛化)
SHORT_PHRASE = {
    "no_helmet": "worker without helmet",
    "no_reflective_vest": "worker without reflective vest",
    "edge_no_guardrail": "unprotected edge",
    "exposed_wiring": "exposed wiring",
}


def sample_instruction(hazard_key, is_sentence=True, rng=random):
    """从模板库采一条指令。"""
    if is_sentence:
        return rng.choice(TEMPLATE_BANK[hazard_key])
    return SHORT_PHRASE[hazard_key]


# ------------------------------------------------------------------ LLM 改写(可选)
def llm_paraphrase(instruction, n=1, client=None):
    """
    可选:调用 LLM 对指令做同义改写,产出更多样的表达。
    未接入 API 时直接返回原句,保证 pipeline 可离线跑通。

    接入示例(Claude / OpenAI 兼容均可):
        prompt = f"把下面这条施工隐患巡检指令,改写成 {n} 条口语化的同义说法,"
                 f"保持意图不变,每行一条:\n{instruction}"
        resp = client.messages.create(...)
        return [line for line in resp_text.splitlines() if line.strip()]
    """
    if client is None:
        return [instruction]
    raise NotImplementedError("接入你的 LLM 客户端后实现改写逻辑")
