# -*- coding: utf-8 -*-
"""
FastText 新闻分类 Streamlit 可视化前端
========================================

基于 Streamlit 的交互式新闻分类预测界面。

功能：
    - 单条预测：输入标题，即时返回分类结果
    - 批量测试：批量输入多条标题，查看预测分布图表
    - 预测历史：保存最近的预测记录，支持清空

启动方式：
    streamlit run api_streamlit.py

与 01_rf、02_xgb 的 api_streamlit.py 的区别：
    - 调用端口改为 8890（FastText 服务）
    - 模型标签：FastText

依赖：
    - streamlit：Web 前端框架
    - requests：调用后端 API
    - pandas：数据展示
"""

import streamlit as st
import requests
import pandas as pd
from collections import Counter
from datetime import datetime

# ============================================================
# 页面配置
# ============================================================
# st.set_page_config() 必须是第一个 Streamlit 命令，用于设置页面元信息。
#   page_title — 浏览器标签页标题
#   page_icon  — 浏览器标签页图标（emoji 或图片路径）
#   layout     — "centered"（居中）或 "wide"（宽屏），宽屏更适合表格展示
st.set_page_config(
    page_title="新闻分类预测（FastText）",
    page_icon="⚡",
    layout="wide",
)

# ============================================================
# 服务地址
# ============================================================
# FastText 服务运行在 8890 端口
API_URL = "http://127.0.0.1:8890/predict_api"
HEALTH_URL = "http://127.0.0.1:8890/health"
REQUEST_TIMEOUT = 10  # 超时秒数，FastText 预测极快，但留足余量

# ============================================================
# 预测函数
# ============================================================
def call_api(title: str) -> dict | None:
    """
    调用 FastText API 进行单条预测。

    参数：
        title : str — 待分类的新闻标题

    返回：
        dict | None — 成功返回 {"title": "...", "pred_class": "..."}
                      失败返回 None
    """
    try:
        resp = requests.post(
            API_URL,
            json={"title": title},
            timeout=REQUEST_TIMEOUT
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# ============================================================
# 侧边栏
# ============================================================
# st.sidebar：侧边栏区域，用于放置导航、状态、参数等
st.sidebar.title("⚡ 新闻分类预测")
st.sidebar.markdown("**模型：** FastText")
st.sidebar.markdown("---")

# 服务状态检查：通过 GET /health 探测服务是否在线
try:
    health = requests.get(HEALTH_URL, timeout=3)
    if health.status_code == 200:
        st.sidebar.success("🟢 API 服务正常运行")
    else:
        st.sidebar.warning("🟡 API 服务状态异常")
except Exception:
    # 服务未启动时，显示启动命令提示
    st.sidebar.error(
        "🔴 API 服务未启动，请先运行：\n"
        "```bash\n"
        "cd 03_fasttext\n"
        "python api_fast_service.py\n"
        "```"
    )

st.sidebar.markdown("---")

# st.radio：单选按钮组，切换不同模式
mode = st.sidebar.radio("选择模式", ["单条预测", "批量测试", "预测历史"])

# ============================================================
# 初始化 session_state
# ============================================================
# st.session_state：Streamlit 的会话状态字典，用于跨页面刷新保存数据。
# 页面每次交互都会重新运行脚本，普通变量会被重置，
# session_state 中的数据会持久化到浏览器会话结束。
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# 模式 1：单条预测
# ============================================================
if mode == "单条预测":
    st.title("📝 单条新闻标题预测")
    st.markdown("输入一条新闻标题，查看 FastText 模型预测的类别。")

    col1, col2 = st.columns([3, 1])
    with col1:
        # st.text_input：单行文本输入框
        #   placeholder — 占位提示文字
        title = st.text_input("新闻标题", placeholder="请输入新闻标题...")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # 微调对齐
        predict_btn = st.button("🔍 预测", type="primary", use_container_width=True)

    if predict_btn and title.strip():
        with st.spinner("预测中..."):
            result = call_api(title.strip())

        if result:
            # st.success：绿色成功提示框
            st.success(f"预测类别：**{result['pred_class']}**")
            # 保存到历史记录
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "title": result["title"],
                "pred_class": result["pred_class"],
            })
        else:
            st.error("预测失败，请检查 API 服务是否启动。")

    # 示例标题：点击即可快速预测
    st.markdown("---")
    st.markdown("**示例标题（点击快速预测）：**")
    examples = [
        "同步A股首秀：港股缩量回调",
        "世界杯决赛明日在卡塔尔开战",
        "最新研究显示人工智能将改变医疗行业",
        "北京房地产新政出台，楼市走向何方",
        "新能源汽车销量持续增长",
    ]
    for ex in examples:
        # st.button：按钮组件，key 保证唯一性
        if st.button(ex, key=f"ex_{ex[:10]}"):
            result = call_api(ex)
            if result:
                st.info(f"**{ex}** → {result['pred_class']}")

# ============================================================
# 模式 2：批量测试
# ============================================================
elif mode == "批量测试":
    st.title("📊 批量测试")
    st.markdown("输入多条标题，每行一条，批量预测并查看分类分布。")

    # st.text_area：多行文本输入框，适合批量输入
    batch_text = st.text_area(
        "批量输入（每行一条标题）",
        height=200,
        placeholder="同步A股首秀：港股缩量回调\n世界杯决赛明日在卡塔尔开战\n..."
    )

    if st.button("🚀 批量预测", type="primary"):
        titles = [t.strip() for t in batch_text.split("\n") if t.strip()]
        if not titles:
            st.warning("请输入至少一条标题。")
        else:
            results = []
            progress = st.progress(0)      # 进度条，0~1
            status = st.empty()            # 占位符，用于动态更新状态文字

            for i, title in enumerate(titles):
                status.text(f"正在预测 {i+1}/{len(titles)}：{title[:30]}...")
                result = call_api(title)
                if result:
                    results.append(result)
                # 更新进度条
                progress.progress((i + 1) / len(titles))

            status.text("预测完成！")
            st.success(f"成功预测 {len(results)}/{len(titles)} 条")

            if results:
                # st.dataframe：交互式表格，支持排序、筛选
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                # 类别分布柱状图
                st.markdown("---")
                st.subheader("类别分布")
                counter = Counter(r["pred_class"] for r in results)
                chart_df = pd.DataFrame(counter.items(), columns=["类别", "数量"])
                chart_df = chart_df.sort_values("数量", ascending=True)
                # st.bar_chart：内置柱状图，自动渲染
                st.bar_chart(chart_df.set_index("类别"))

# ============================================================
# 模式 3：预测历史
# ============================================================
elif mode == "预测历史":
    st.title("📋 预测历史")
    if not st.session_state.history:
        st.info("暂无预测记录。")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

        if st.button("🗑️ 清空历史"):
            # 清空 session_state 并重新运行页面
            st.session_state.history = []
            st.rerun()