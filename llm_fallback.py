"""
=============================================================================
llm_fallback.py —— LLM 兜底分类器（低置信度时调用）
=============================================================================
【文件作用】
    当传统模型预测置信度低于阈值时，调用 LLM 进行兜底分类。

【调用方式】
    from llm_fallback import llm_classify
    result = llm_classify("一条新闻标题")

【环境变量】
    DASHSCOPE_API_KEY — 阿里云 DashScope API Key（必须）
    LLM_MODEL_NAME     — 模型名称（可选，默认 qwen3.8-flash）

【类别限制】
    LLM 只能从以下 10 个类别中选择：
    0: finance, 1: realty, 2: stocks, 3: education, 4: science,
    5: society, 6: politics, 7: sports, 8: game, 9: entertainment

【返回格式（统一）】
    {
        "label_id": int,
        "label": str,
        "confidence": None,
        "source": "llm"
    }
=============================================================================
"""
import os
import json
from openai import OpenAI


# ==========================================================================
# 类别定义（与 class.txt 一致，索引 0-9）
# ==========================================================================
CLASS_DICT = {
    0: "finance",
    1: "realty",
    2: "stocks",
    3: "education",
    4: "science",
    5: "society",
    6: "politics",
    7: "sports",
    8: "game",
    9: "entertainment",
}

# 反向映射：用于校验 LLM 返回值
_NAME_TO_ID = {v: k for k, v in CLASS_DICT.items()}


# ==========================================================================
# LLM 配置
# ==========================================================================
_LLM_MODEL = os.environ.get("LLM_MODEL_NAME", "qwen3.8-flash")
_LLM_BASE_URL = "https://ws-rt3ln75swgtdisjl.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


def _get_client():
    """获取 LLM 客户端（每次调用时检查 API Key）"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "未设置 DASHSCOPE_API_KEY 环境变量。\n"
            "请在 .env 文件中设置或通过系统环境变量设置：\n"
            "  set DASHSCOPE_API_KEY=your_key_here"
        )
    return OpenAI(api_key=api_key, base_url=_LLM_BASE_URL)


# ==========================================================================
# LLM 分类
# ==========================================================================
_SYSTEM_PROMPT = """你是一个新闻标题分类器。请对输入的新闻标题进行分类。

你只能从以下10个类别中选择一个：
0: finance（财经）
1: realty（房产）
2: stocks（股票）
3: education（教育）
4: science（科技）
5: society（社会）
6: politics（政治）
7: sports（体育）
8: game（游戏）
9: entertainment（娱乐）

请严格返回JSON格式，不要输出任何解释、分析或其他内容：
{"label_id": <类别编号>, "label": "<类别英文名>"}"""


def llm_classify(text: str) -> dict:
    """
    调用 LLM 对新闻标题进行兜底分类。

    参数:
        text : str — 新闻标题文本

    返回:
        dict — {
            "label_id": int,
            "label": str,
            "confidence": None,
            "source": "llm"
        }

    注意:
        - LLM 不提供 confidence（设为 None，不伪造）
        - 如果 LLM 返回的类别不在 10 类中，label_id 设为 -1
    """
    client = _get_client()

    print(f"[LLM Fallback] 调用模型: {_LLM_MODEL}")
    print(f"[LLM Fallback] 输入文本: {text[:60]}...")

    response = client.chat.completions.create(
        model=_LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=100,
    )

    raw_output = response.choices[0].message.content
    print(f"[LLM Fallback] LLM 原始返回: {raw_output}")

    result = json.loads(raw_output)

    label = result.get("label", "unknown")
    label_id = result.get("label_id", -1)

    # 校验：如果 LLM 返回了类别名而不是编号，尝试转换
    if isinstance(label_id, str):
        label_id = _NAME_TO_ID.get(label_id, -1)

    # 确保 label_id 在有效范围内
    label_id = int(label_id)
    if label_id not in CLASS_DICT:
        label = "unknown"
        label_id = -1

    # 确保 label 和 label_id 一致
    if label_id >= 0 and label in _NAME_TO_ID:
        pass  # 一致，无需处理
    elif label_id >= 0:
        label = CLASS_DICT[label_id]

    print(f"[LLM Fallback] 最终分类: label_id={label_id}, label={label}")

    return {
        "label_id": label_id,
        "label": label,
        "confidence": None,
        "source": "llm",
    }


# ==========================================================================
# 测试入口
# ==========================================================================
if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "NBA季后赛湖人队逆转取胜"
    result = llm_classify(text)
    print(f"\n结果: {result}")