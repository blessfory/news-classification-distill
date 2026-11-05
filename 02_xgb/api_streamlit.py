# -*- coding: utf-8 -*-
"""
XGBoost 新闻分类 Streamlit 可视化前端
======================================

基于 Streamlit 的交互式新闻分类预测界面。

功能：
    - 单条预测：输入标题，即时返回分类结果
    - 批量测试：批量输入多条标题，查看预测分布
    - 历史记录：保存最近的预测记录

启动方式：
    streamlit run api_streamlit.py

与 01_rf 的 api_streamlit.py 的区别：
    - 调用端口改为 8889（XGBoost 服务）
"""

import streamlit as st
import requests
import pandas as pd
from collections import Counter
from datetime import datetime

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="新闻分类预测（XGBoost）",
    page_icon="📰",
    layout="wide",
)

# ============================================================
# 服务地址
# ============================================================
API_URL = "http://127.0.0.1:8889/predict_api"
HEALTH_URL = "http://127.0.0.1:8889/health"
REQUEST_TIMEOUT = 10

# ============================================================
# 预测函数
# ============================================================
def call_api(title: str) -> dict | None:
    """调用 XGBoost API 进行预测"""
    try:
        resp = requests.post(API_URL, json={"title": title}, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# ============================================================
# 侧边栏
# ============================================================
st.sidebar.title("📰 新闻分类预测")
st.sidebar.markdown("**模型：** Word2Vec + XGBoost")
st.sidebar.markdown("---")

# 服务状态检查
try:
    health = requests.get(HEALTH_URL, timeout=3)
    if health.status_code == 200:
        st.sidebar.success("🟢 API 服务正常运行")
    else:
        st.sidebar.warning("🟡 API 服务状态异常")
except Exception:
    st.sidebar.error("🔴 API 服务未启动，请先运行：\n```bash\npython api_fast_service.py\n```")

st.sidebar.markdown("---")
mode = st.sidebar.radio("选择模式", ["单条预测", "批量测试", "预测历史"])

# ============================================================
# 初始化 session_state
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# 模式 1：单条预测
# ============================================================
if mode == "单条预测":
    st.title("📝 单条新闻标题预测")
    st.markdown("输入一条新闻标题，查看模型预测的类别。")

    col1, col2 = st.columns([3, 1])
    with col1:
        title = st.text_input("新闻标题", placeholder="请输入新闻标题...")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 预测", type="primary", use_container_width=True)

    if predict_btn and title.strip():
        with st.spinner("预测中..."):
            result = call_api(title.strip())

        if result:
            st.success(f"预测类别：**{result['pred_class']}**")
            # 保存到历史
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "title": result["title"],
                "pred_class": result["pred_class"],
            })
        else:
            st.error("预测失败，请检查 API 服务是否启动。")

    # 示例数据
    st.markdown("---")
    st.markdown("**示例标题（点击复制）：**")
    examples = [
        "同步A股首秀：港股缩量回调",
        "世界杯决赛明日在卡塔尔开战",
        "最新研究显示人工智能将改变医疗行业",
        "北京房地产新政出台，楼市走向何方",
        "新能源汽车销量持续增长",
    ]
    for ex in examples:
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
            progress = st.progress(0)
            status = st.empty()

            for i, title in enumerate(titles):
                status.text(f"正在预测 {i+1}/{len(titles)}：{title[:30]}...")
                result = call_api(title)
                if result:
                    results.append(result)
                progress.progress((i + 1) / len(titles))

            status.text("预测完成！")
            st.success(f"成功预测 {len(results)}/{len(titles)} 条")

            if results:
                # 表格展示
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                # 类别分布图
                st.markdown("---")
                st.subheader("类别分布")
                counter = Counter(r["pred_class"] for r in results)
                chart_df = pd.DataFrame(counter.items(), columns=["类别", "数量"])
                chart_df = chart_df.sort_values("数量", ascending=True)
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
            st.session_state.history = []
            st.rerun()