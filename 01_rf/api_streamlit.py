# -*- coding: utf-8 -*-
"""
FastAPI 新闻分类预测 —— Streamlit 前端
========================================

与 api_fast_service.py 配套使用，提供浏览器端的可视化预测界面。

使用步骤：
    1. 先启动服务端：python api_fast_service.py
       （或 uvicorn api_fast_service:app --host 127.0.0.1 --port 8888 --reload）
    2. 再运行本文件：streamlit run api_streamlit.py
    3. 浏览器自动打开 http://localhost:8501

功能特性：
    - 单条新闻标题预测
    - 批量预测（支持示例快速测试）
    - 预测历史记录与类别分布统计
    - 服务端健康检查
"""

import time
import requests
import pandas as pd
import streamlit as st

# ============================================================
# 页面配置（必须是第一条 st 命令）
# ============================================================
st.set_page_config(
    page_title="新闻分类预测系统",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 类别映射（英文标签 → 中文显示名 + emoji 图标）
# ============================================================
CATEGORY_MAP = {
    "finance":       ("财经",    "💰"),
    "realty":        ("房产",    "🏠"),
    "stocks":        ("股票",    "📈"),
    "education":     ("教育",    "🎓"),
    "science":       ("科技",    "🔬"),
    "society":       ("社会",    "👥"),
    "politics":      ("政治",    "🏛️"),
    "sports":        ("体育",    "⚽"),
    "game":          ("游戏",    "🎮"),
    "entertainment": ("娱乐",    "🎬"),
}

# ============================================================
# 示例新闻标题（用于快速测试）
# ============================================================
EXAMPLE_TITLES = [
    "同步A股首秀：港股缩量回调",
    "房地产市场新政出台 多地房价应声下跌",
    "苹果发布新款iPhone 搭载最新芯片",
    "人工智能大模型再创新高 科技股集体涨停",
    "体育总局发布新规 足球联赛即将开战",
    "高考志愿填报开启 名校竞争激烈",
    "嫦娥六号成功着陆月球背面 传回首张图像",
    "中央经济工作会议召开 部署来年重点任务",
    "英雄联盟S14总决赛落幕 冠军战队诞生",
    "知名导演新片上映 票房首日破亿",
]

# ============================================================
# Streamlit 会话状态初始化
# ============================================================
if "history" not in st.session_state:
    # 存储预测历史，每条为 {"title": str, "pred_class": str, "time": str}
    st.session_state.history = []


# ============================================================
# 工具函数
# ============================================================
def get_api_base_url():
    """从侧边栏读取 API 地址"""
    return st.session_state.get("api_url", "http://127.0.0.1:8888")


def check_server():
    """
    检查服务端是否可用（调用 /health 健康检查接口）

    Returns:
        tuple: (是否可用, 服务端信息字典或错误信息)
    """
    base_url = get_api_base_url()
    try:
        resp = requests.get(f"{base_url}/health", timeout=3)
        if resp.status_code == 200:
            return True, resp.json()
        return False, f"HTTP {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "连接被拒绝，请确认服务端已启动"
    except requests.exceptions.Timeout:
        return False, "请求超时"
    except requests.exceptions.RequestException as e:
        return False, str(e)


def predict_title(title):
    """
    调用预测接口，对单条新闻标题进行分类

    Args:
        title: 待分类的新闻标题

    Returns:
        dict or None: 预测结果字典；请求失败时返回 None
    """
    base_url = get_api_base_url()
    payload = {"title": title}

    try:
        resp = requests.post(f"{base_url}/predict_api", json=payload, timeout=5)

        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 422:
            st.error(f"参数校验失败（422）：{resp.json()}")
        else:
            st.error(f"服务端错误（{resp.status_code}）：{resp.text}")

    except requests.exceptions.ConnectionError:
        st.error("连接被拒绝，请先启动服务端 `python api_fast_service.py`")
    except requests.exceptions.Timeout:
        st.error("请求超时，请检查服务端是否正常运行")
    except requests.exceptions.RequestException as e:
        st.error(f"请求异常：{e}")

    return None


def get_category_display(pred_class):
    """
    获取类别的中文显示名和 emoji 图标

    Args:
        pred_class: 英文类别名（如 'finance'）

    Returns:
        tuple: (中文显示名, emoji 图标)
    """
    if pred_class in CATEGORY_MAP:
        return CATEGORY_MAP[pred_class]
    return pred_class, "❓"


def add_to_history(title, pred_class):
    """
    将预测结果追加到会话历史

    Args:
        title: 新闻标题
        pred_class: 预测类别
    """
    st.session_state.history.append({
        "title": title,
        "pred_class": pred_class,
        "time": time.strftime("%H:%M:%S"),
    })


# ============================================================
# 侧边栏：配置面板
# ============================================================
with st.sidebar:
    st.header("⚙️ 配置面板")

    # API 地址配置
    api_url = st.text_input(
        "API 服务地址",
        value="http://127.0.0.1:8888",
        help="FastAPI 服务端地址",
    )
    st.session_state.api_url = api_url

    st.divider()

    # 服务状态检查
    st.subheader("🔗 服务状态")
    server_ok, server_info = check_server()

    if server_ok:
        st.success(f"✅ 已连接\n\n{server_info}")
    else:
        st.error(f"❌ 未连接\n\n{server_info}")
        st.info(
            "💡 **启动服务端**：\n\n"
            "```bash\n"
            "cd 01_rf\n"
            "python api_fast_service.py\n"
            "```"
        )

    st.divider()

    # 操作说明
    st.subheader("📖 使用说明")
    st.markdown(
        """
        1. **单条预测**：在"单条预测"标签页输入新闻标题
        2. **批量预测**：在"批量预测"标签页粘贴多条标题
        3. **历史记录**：在"历史记录"标签页查看预测历史和统计
        """
    )

    st.divider()

    # 类别对照表
    st.subheader("🏷️ 类别对照表")
    for eng, (cn, emoji) in CATEGORY_MAP.items():
        st.markdown(f"{emoji} **{cn}** (`{eng}`)")


# ============================================================
# 主标题
# ============================================================
st.title("📰 新闻分类预测系统")
st.caption("基于 TF-IDF + 随机森林的中文新闻标题分类 · FastAPI + Streamlit")

# ============================================================
# 标签页
# ============================================================
tab_single, tab_batch, tab_history = st.tabs([
    "🎯 单条预测",
    "📦 批量预测",
    "📊 历史记录",
])

# ============================================================
# Tab1：单条预测
# ============================================================
with tab_single:
    st.subheader("单条新闻标题预测")

    # 文本输入
    user_input = st.text_area(
        "请输入新闻标题",
        value="",
        height=100,
        placeholder="例如：同步A股首秀：港股缩量回调",
    )

    # 快捷示例
    st.caption("💡 点击下方示例快速填充：")
    cols = st.columns(3)
    for i, title in enumerate(EXAMPLE_TITLES[:6]):
        with cols[i % 3]:
            if st.button(f"📝 {title[:15]}...", key=f"example_{i}"):
                st.session_state["single_input"] = title

    # 如果有快捷按钮填入的内容
    if "single_input" in st.session_state:
        user_input = st.session_state.pop("single_input")

    # 预测按钮
    if st.button("🚀 开始预测", type="primary", disabled=not check_server()[0]):
        if not user_input.strip():
            st.warning("⚠️ 请输入新闻标题后再提交")
        else:
            with st.spinner("🔄 正在预测中..."):
                result = predict_title(user_input.strip())

            if result:
                pred_class = result["pred_class"]
                cn_name, emoji = get_category_display(pred_class)

                # 大结果展示
                st.success(f"{emoji} 预测类别：**{cn_name}** (`{pred_class}`)")
                st.info(f"📰 原始标题：{result['title']}")

                # 保存到历史
                add_to_history(result["title"], pred_class)
            else:
                st.error("❌ 预测失败，请检查服务端状态")

# ============================================================
# Tab2：批量预测
# ============================================================
with tab_batch:
    st.subheader("批量新闻标题预测")
    st.caption("每行一条新闻标题，支持粘贴多条同时预测")

    batch_input = st.text_area(
        "请输入新闻标题（每行一条）",
        height=200,
        placeholder="每行输入一条新闻标题...",
    )

    # 预置示例数据
    st.caption("💡 使用下方示例快速填充：")
    example_text = "\n".join(EXAMPLE_TITLES)
    if st.button("📋 填充示例数据"):
        st.session_state["batch_input"] = example_text

    if "batch_input" in st.session_state:
        batch_input = st.session_state.pop("batch_input")

    if st.button("🚀 开始批量预测", type="primary", disabled=not check_server()[0]):
        titles = [line.strip() for line in batch_input.split("\n") if line.strip()]

        if not titles:
            st.warning("⚠️ 请至少输入一条新闻标题")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, title in enumerate(titles):
                status_text.text(f"🔄 正在预测第 {i + 1}/{len(titles)} 条：{title[:20]}...")
                result = predict_title(title)
                if result:
                    results.append(result)
                    add_to_history(result["title"], result["pred_class"])
                else:
                    results.append({"title": title, "pred_class": "❌ 预测失败"})
                progress_bar.progress((i + 1) / len(titles))
                time.sleep(0.1)  # 轻微延时，避免请求过快

            status_text.text("✅ 批量预测完成！")

            # 结果表格
            st.subheader("📋 预测结果")
            df_results = pd.DataFrame(results)
            df_results["序号"] = range(1, len(df_results) + 1)
            df_results["类别"] = df_results["pred_class"].apply(
                lambda x: f"{get_category_display(x)[1]} {get_category_display(x)[0]}"
                if x in CATEGORY_MAP else x
            )
            df_results = df_results[["序号", "title", "类别", "pred_class"]]
            df_results.columns = ["序号", "新闻标题", "类别（中文）", "类别（英文）"]
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            # 下载结果
            csv = df_results.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 下载结果 CSV",
                data=csv,
                file_name=f"predictions_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

# ============================================================
# Tab3：历史记录与统计
# ============================================================
with tab_history:
    st.subheader("📊 预测历史与统计")

    if not st.session_state.history:
        st.info("📭 暂无历史预测记录，请先到'单条预测'或'批量预测'标签页进行预测")
    else:
        history = st.session_state.history

        # 统计信息
        col1, col2, col3 = st.columns(3)
        col1.metric("总预测数", len(history))
        col2.metric("涉及类别", len(set(h["pred_class"] for h in history)))

        # 类别分布统计
        class_counts = {}
        for h in history:
            cls = h["pred_class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        col3.metric("最多类别", max(class_counts, key=class_counts.get))

        st.divider()

        # 类别分布柱状图
        st.subheader("🏷️ 类别分布")
        df_dist = pd.DataFrame([
            {"类别": get_category_display(k)[0], "数量": v}
            for k, v in class_counts.items()
        ]).sort_values("数量", ascending=False)
        st.bar_chart(df_dist.set_index("类别"))

        st.divider()

        # 历史记录表格
        st.subheader("📋 历史预测详情")
        df_history = pd.DataFrame(history)
        df_history.columns = ["时间", "新闻标题", "预测类别"]
        df_history["时间"] = [h["time"] for h in history]
        df_history["类别（中文）"] = df_history["预测类别"].apply(
            lambda x: f"{get_category_display(x)[1]} {get_category_display(x)[0]}"
            if x in CATEGORY_MAP else x
        )
        df_history = df_history[["时间", "新闻标题", "类别（中文）", "预测类别"]]
        df_history.columns = ["时间", "新闻标题", "类别（中文）", "类别（英文）"]
        st.dataframe(df_history, use_container_width=True, hide_index=True)

        # 清空历史
        st.divider()
        if st.button("🗑️ 清空历史记录"):
            st.session_state.history = []
            st.success("✅ 历史记录已清空")
            st.rerun()