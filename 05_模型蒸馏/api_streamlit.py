"""
====================================================================================================
api_streamlit.py —— 知识蒸馏模型 Streamlit 可视化前端
====================================================================================================
【快速启动】
    streamlit run api_streamlit.py
====================================================================================================
"""
import streamlit as st
import requests
import time
import pandas as pd

API_URL = "http://127.0.0.1:8892"

# ==========================================================================
# 页面配置
# ==========================================================================
st.set_page_config(
    page_title="知识蒸馏 - 新闻分类",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 知识蒸馏模型 - 新闻分类预测")
st.markdown("---")

# ==========================================================================
# 侧边栏：模型选择 + 信息
# ==========================================================================
with st.sidebar:
    st.header("⚙️ 模型配置")

    model_type = st.selectbox(
        "选择模型",
        options=['bilstm', 'transformer', 'teacher', 'prune'],
        format_func=lambda x: {
            'bilstm': 'BiLSTM 学生模型（约 9.6M）',
            'transformer': 'Transformer 学生模型（约 15M）',
            'teacher': 'BERT 教师模型（约 110M）',
            'prune': '剪枝后 BERT 模型'
        }[x]
    )

    # 检查服务
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        if resp.status_code == 200:
            st.success("✅ 服务连接成功")
        else:
            st.error("❌ 服务异常")
    except:
        st.error("❌ 服务未启动\n请运行: python api_fast_service.py")

    # 获取模型信息
    try:
        info = requests.get(f"{API_URL}/model_info", params={"model_type": model_type}, timeout=5).json()
        st.markdown("---")
        st.markdown(f"**参数量:** {info['params']:,}")
        st.markdown(f"**类别数:** {info['class_cnt']}")
        st.markdown(f"**设备:** {info['device']}")
    except:
        pass

    st.markdown("---")
    st.caption("模型蒸馏流程：BERT → 剪枝 → 蒸馏 → 学生模型")

# ==========================================================================
# 主区域：预测
# ==========================================================================
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📝 单条预测")
    text_input = st.text_area(
        "输入新闻标题",
        placeholder="例如：NBA季后赛湖人队大胜勇士",
        height=100
    )

    if st.button("🔮 开始预测", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("请输入文本")
        else:
            with st.spinner("预测中..."):
                start = time.perf_counter()
                try:
                    resp = requests.post(
                        f"{API_URL}/predict_api",
                        json={"text": text_input, "model_type": model_type},
                        timeout=30
                    )
                    elapsed = (time.perf_counter() - start) * 1000

                    if resp.status_code == 200:
                        result = resp.json()

                        st.success(f"预测类别: **{result['class']}** | 置信度: **{result['confidence']:.4f}** | 耗时: {elapsed:.1f}ms")

                        # 概率分布表
                        prob_data = []
                        for item in result['probabilities']:
                            prob_data.append({
                                '类别': item['class'],
                                '概率': item['probability']
                            })
                        df = pd.DataFrame(prob_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)

                    else:
                        st.error(f"预测失败: {resp.text}")
                except Exception as e:
                    st.error(f"请求失败: {e}")

with col2:
    st.subheader("📊 批量测试")
    st.caption("用预设测试集快速验证模型效果")

    preset_tests = [
        "NBA季后赛湖人队大胜勇士",
        "央行宣布降准释放流动性",
        "新iPhone发布引发抢购热潮",
        "教育部发布双减政策实施细则",
        "世界杯决赛法国队夺冠",
    ]

    if st.button("🚀 批量测试", use_container_width=True):
        results = []
        total_time = 0

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, text in enumerate(preset_tests):
            status_text.text(f"测试中: {text[:20]}...")
            start = time.perf_counter()
            try:
                resp = requests.post(
                    f"{API_URL}/predict_api",
                    json={"text": text, "model_type": model_type},
                    timeout=30
                )
                elapsed = (time.perf_counter() - start) * 1000
                total_time += elapsed

                if resp.status_code == 200:
                    result = resp.json()
                    results.append({
                        '标题': text,
                        '预测类别': result['class'],
                        '置信度': f"{result['confidence']:.4f}",
                        '耗时(ms)': f"{elapsed:.1f}"
                    })
            except:
                results.append({'标题': text, '预测类别': '错误', '置信度': '-', '耗时(ms)': '-'})

            progress_bar.progress((i + 1) / len(preset_tests))

        status_text.empty()

        if results:
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, hide_index=True)
            avg_time = total_time / len(preset_tests)
            st.info(f"平均耗时: {avg_time:.1f} ms/条")

# ==========================================================================
# 底部说明
# ==========================================================================
st.markdown("---")
st.caption("知识蒸馏：先剪枝（10% 权重置零）→ 后蒸馏（教师软标签指导）→ 学生模型部署")