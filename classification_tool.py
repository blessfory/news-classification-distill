"""
=============================================================================
classification_tool.py —— 文本分类 Tool（供 Agent 调用）
=============================================================================
【文件作用】
    把现有的 predict_with_fallback() 包装成 LangChain Tool，
    供 Agent 通过 function calling 调用。

【Tool 说明】
    - 输入：text（新闻标题）
    - 输出：统一分类结果 {label_id, label, confidence, source}
    - 内部使用 XGBoost 作为默认模型（已验证最稳定）

【调用方式（独立使用）】
    from classification_tool import classify_text
    result = classify_text.invoke({"text": "一条新闻标题"})
=============================================================================
"""
from langchain_core.tools import tool

from unified_predict import predict_with_fallback

# 默认：XGBoost 模型，阈值 0.6
_DEFAULT_MODEL = "xgb"
_DEFAULT_THRESHOLD = 0.6


@tool
def classify_text(text: str) -> dict:
    """
    对新闻标题进行文本分类。

    类别包括：finance(财经), realty(房产), stocks(股票), education(教育),
    science(科技), society(社会), politics(政治), sports(体育), game(游戏),
    entertainment(娱乐)。

    参数:
        text: 需要分类的新闻标题文本

    返回:
        dict — {
            "label_id":   类别编号（0-9），
            "label":      类别英文名，
            "confidence": 置信度（float 或 None），
            "source":     "model"（传统模型）或 "llm"（LLM兜底）
        }
    """
    return predict_with_fallback(
        text,
        model_name=_DEFAULT_MODEL,
        threshold=_DEFAULT_THRESHOLD,
    )