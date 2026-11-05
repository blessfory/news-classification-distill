"""
=============================================================================
agent.py —— LLM Agent 文本分类助手
=============================================================================
【文件作用】
    创建 LLM Agent，包装 classification_tool，提供 agent_predict() 入口。

【Agent 职责】
    1. 接收用户输入的新闻标题
    2. 调用 classify_text Tool 完成分类
    3. 整理 Tool 返回结果，向用户说明分类结果和来源

    分类逻辑由 unified_predict → predict_with_fallback → Tool 负责，
    Agent 只负责调度和整理结果，不自行判断类别。

【调用方式】
    from agent import agent_predict
    result = agent_predict("NBA季后赛湖人队大胜勇士")

【环境变量】
    DASHSCOPE_API_KEY — 阿里云 DashScope API Key（必须）
    LLM_MODEL_NAME     — Agent 使用的模型名称（可选，默认 qwen3.8-flash）
=============================================================================
"""
import os

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from classification_tool import classify_text


# ==========================================================================
# LLM 配置（与 llm_fallback.py 共用同一套环境变量）
# ==========================================================================
_LLM_MODEL = os.environ.get("LLM_MODEL_NAME", "qwen3.8-flash")
_LLM_BASE_URL = "https://ws-rt3ln75swgtdisjl.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


def _get_llm():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "未设置 DASHSCOPE_API_KEY 环境变量。\n"
            "请在 .env 文件中设置或通过系统环境变量设置。\n"
            "  set DASHSCOPE_API_KEY=your_key_here"
        )
    return ChatOpenAI(
        model=_LLM_MODEL,
        api_key=api_key,
        base_url=_LLM_BASE_URL,
        temperature=0,
    )


# ==========================================================================
# System Prompt
# ==========================================================================
_SYSTEM_PROMPT = """你是一个文本分类助手。

当用户提供需要分类的文本时，调用 classify_text 工具完成分类。

规则：
- 不要自己猜测分类结果。
- 不要自己生成不存在的类别。
- 分类结果必须以工具返回结果为准。
- 如果工具返回 source="model"，说明由传统分类模型直接判断。
- 如果工具返回 source="llm"，说明传统模型置信度较低，由 LLM 进行了兜底分类。

向用户输出分类结果时，请说明：
1. 分类类别（中文译名）
2. 类别 ID
3. 分类来源（传统模型 / LLM兜底）
4. 置信度（如果有）"""


# ==========================================================================
# Agent 创建（单例，避免重复初始化）
# ==========================================================================
_AGENT = None


def _get_agent():
    global _AGENT
    if _AGENT is not None:
        return _AGENT

    llm = _get_llm()
    tools = [classify_text]

    _AGENT = create_agent(llm, tools, system_prompt=_SYSTEM_PROMPT)
    return _AGENT


# ==========================================================================
# 预测入口
# ==========================================================================
def agent_predict(text: str) -> str:
    """
    Agent 预测入口。

    参数:
        text : str — 新闻标题文本

    返回:
        str — Agent 的最终回答（包含分类结果及说明）
    """
    agent = _get_agent()
    from langchain_core.messages import HumanMessage
    result = agent.invoke({"messages": [HumanMessage(content=text)]})
    messages = result["messages"]
    return messages[-1].content


# ==========================================================================
# 测试入口
# ==========================================================================
if __name__ == "__main__":
    import sys

    text = sys.argv[1] if len(sys.argv) > 1 else "NBA季后赛湖人队大胜勇士"
    print(f"用户输入: {text}")
    print("=" * 60)
    answer = agent_predict(text)
    print("=" * 60)
    print(f"\nAgent 回答:\n{answer}")