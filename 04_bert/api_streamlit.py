"""
====================================================================================================
api_streamlit.py —— BERT 新闻分类 Streamlit 可视化前端
====================================================================================================
【文件作用】
    基于 Streamlit 的交互式新闻分类预测界面，通过 HTTP 调用 BERT API 服务。

【快速启动】
    streamlit run api_streamlit.py

【功能模块】
    - 单条预测：输入标题，即时返回分类结果
    - 批量测试：批量输入多条标题，查看预测分布图表
    - 预测历史：保存最近的预测记录，支持清空

【依赖包详细说明】
----------------------------------------------------------------------------------------------------
1. streamlit
   极简的 Python Web 前端框架，专为数据科学/机器学习场景设计。
   不需要 HTML/CSS/JS 知识，纯 Python 代码即可构建交互式界面。

   常用 API：
   - st.set_page_config(page_title, page_icon, layout)  页面配置（必须是第一个 Streamlit 命令）
       参数：
         page_title: 浏览器标签页标题
         page_icon:  标签页图标（emoji 或图片路径）
         layout:     "centered"（居中）或 "wide"（宽屏）

   - st.title(text)         页面标题（H1）
   - st.markdown(text)      渲染 Markdown 文本
   - st.text_input(label, placeholder)  单行文本输入框
   - st.text_area(label, height, placeholder)  多行文本输入框
   - st.button(label, type, key)  按钮
       参数：
         type: "primary"（主按钮，蓝色）, "secondary"（次要按钮，灰色）
         key:  唯一标识，防止重复点击冲突
   - st.radio(label, options)  单选按钮组
   - st.success(message)   绿色成功提示
   - st.error(message)     红色错误提示
   - st.warning(message)   黄色警告提示
   - st.info(message)      蓝色信息提示
   - st.spinner(text)      加载动画（with 上下文管理器）
   - st.progress(value)    进度条（0.0~1.0）
   - st.empty()            占位符，用于动态更新内容
   - st.dataframe(df, use_container_width)  交互式表格
   - st.bar_chart(data)    柱状图
   - st.columns(ratios)    多列布局，返回各列的容器
   - st.sidebar            侧边栏对象，可放置导航和控件
   - st.session_state      会话状态字典，跨页面刷新保存数据
   - st.rerun()            重新运行整个脚本

2. requests
   调用后端 API 的 HTTP 客户端库。
   详见 api_local_client.py 中的说明。

3. pandas
   数据处理库，用于表格展示和图表数据准备。
   常用 API：
   - pd.DataFrame(data)  从字典/列表创建 DataFrame
   - df.sort_values(by, ascending)  按列排序

4. collections.Counter
   Python 标准库，用于统计元素频率。
   常用 API：
   - Counter(iterable)             创建计数器
   - Counter.most_common(n)        返回频率最高的 n 个元素
   - Counter.items()               返回 (元素, 计数) 的键值对

5. datetime
   Python 标准库，日期时间处理。
   常用 API：
   - datetime.now()                         当前时间
   - datetime.now().strftime("%H:%M:%S")    格式化为字符串

【面试高频考点 ⭐】
    1. Streamlit 和传统 Web 框架（Flask/Django）的区别？
    2. st.session_state 的作用？（跨页面刷新保持数据，类似浏览器的 session）
    3. Streamlit 的页面刷新机制？（每次交互重新运行整个脚本）
    4. requests.post 的 json 参数和 data 参数的区别？
    5. Python 的 Counter 类用法？（统计元素频率）
    6. 列表推导式的写法和效率？
====================================================================================================
"""
import streamlit as st
import requests
import pandas as pd
from collections import Counter
from datetime import datetime

# ==========================================================================
# 页面配置
# ==========================================================================
# st.set_page_config() 必须是第一个 Streamlit 命令，用于设置页面元信息。
#   page_title — 浏览器标签页标题
#   page_icon  — 浏览器标签页图标（emoji）
#   layout     — "wide" 宽屏布局，更适合表格和图表展示
st.set_page_config(
    page_title="新闻分类预测（BERT）",
    page_icon="🤖",
    layout="wide",
)

# ==========================================================================
# 服务地址
# ==========================================================================
# BERT 服务运行在 8891 端口
# 端口分配：01_rf=8888, 02_xgb=8889, 03_fasttext=8890, 04_bert=8891
API_URL = "http://127.0.0.1:8891/predict_api"
HEALTH_URL = "http://127.0.0.1:8891/health"
REQUEST_TIMEOUT = 15  # BERT 预测较慢，适当放宽超时时间


# ==========================================================================
# 预测函数：调用 BERT API 进行单条预测
# ==========================================================================
def call_api(title: str) -> dict | None:
    """
    调用 BERT API 进行单条预测。

    参数：
        title : str — 待分类的新闻标题

    返回：
        dict | None — 成功返回 {"title": "...", "pred_class": "..."}
                      失败返回 None

    异常处理：
        所有异常统一捕获，返回 None。
        原因：Streamlit 前端不需要细粒度错误，只需知道"预测失败"即可。
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


# ==========================================================================
# 侧边栏
# ==========================================================================
# st.sidebar：侧边栏区域，用于放置导航、状态、说明等
st.sidebar.title("🤖 新闻分类预测")
st.sidebar.markdown("**模型：** BERT（预训练语言模型）")
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
        "cd 04_bert\n"
        "python api_fast_service.py\n"
        "```"
    )

st.sidebar.markdown("---")

# BERT 模型说明
st.sidebar.markdown("""
**关于 BERT 模型：**
- 基于 Transformer 架构的预训练语言模型
- 中文版本：bert-base-chinese
- 采用特征提取模式（冻结 BERT 参数）
- 预测速度：GPU 约 10~50ms，CPU 约 100~500ms

**与其他模型对比：**
| 模型 | 端口 | 速度 | 精度 |
|------|------|------|------|
| 随机森林 | 8888 | 极快 | 一般 |
| XGBoost | 8889 | 快 | 较好 |
| FastText | 8890 | 快 | 较好 |
| BERT | 8891 | 较慢 | 最高 |
""")

st.sidebar.markdown("---")

# st.radio：单选按钮组，切换不同功能模式
mode = st.sidebar.radio("选择模式", ["单条预测", "批量测试", "预测历史"])

# ==========================================================================
# 初始化 session_state
# ==========================================================================
# st.session_state：Streamlit 的会话状态字典，用于跨页面刷新保存数据。
# 页面每次交互都会重新运行脚本，普通变量会被重置，
# session_state 中的数据会持久化到当前浏览器会话结束。
if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================================================
# 模式 1：单条预测
# ==========================================================================
if mode == "单条预测":
    st.title("📝 单条新闻标题预测")
    st.markdown("输入一条新闻标题，查看 BERT 模型预测的类别。")

    # st.columns：创建多列布局，方便水平排列控件
    # [3, 1] 表示第一列占 3/4 宽度，第二列占 1/4 宽度
    col1, col2 = st.columns([3, 1])
    with col1:
        # st.text_input：单行文本输入框
        #   placeholder — 占位提示文字，显示在输入框内
        title = st.text_input("新闻标题", placeholder="请输入新闻标题...")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # 用空行微调垂直对齐
        # st.button：按钮，type="primary" 为蓝色主按钮，use_container_width 填满列宽
        predict_btn = st.button("🔍 预测", type="primary", use_container_width=True)

    if predict_btn and title.strip():
        # st.spinner：加载动画，with 上下文管理器自动显示/隐藏
        with st.spinner("BERT 模型预测中，请稍候..."):
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

    # 示例标题：点击即可快速预测，方便演示
    st.markdown("---")
    st.markdown("**示例标题（点击快速预测）：**")
    examples = [
        "近期新盘推荐 通州纯新别墅本周开盘",
        "NBA季后赛 湖人队逆转取胜",
        "最新研究显示人工智能将改变医疗行业",
        "北京房地产新政出台，楼市走向何方",
        "新能源汽车销量持续增长",
        "教育部发布双减政策最新进展",
        "苹果发布新款iPhone 搭载最新芯片",
        "世界杯决赛明日在卡塔尔开战",
    ]
    # 每行显示 4 个示例按钮
    cols = st.columns(4)
    for i, ex in enumerate(examples):
        with cols[i % 4]:
            # st.button：每个按钮需要唯一 key，防止 Streamlit 报重复错误
            if st.button(ex, key=f"ex_{i}"):
                with st.spinner("预测中..."):
                    result = call_api(ex)
                if result:
                    st.info(f"**{ex}** → {result['pred_class']}")


# ==========================================================================
# 模式 2：批量测试
# ==========================================================================
elif mode == "批量测试":
    st.title("📊 批量测试")
    st.markdown("输入多条标题，每行一条，批量预测并查看分类分布。")

    # st.text_area：多行文本输入框，适合批量输入
    #   height=200 — 输入框高度（像素）
    batch_text = st.text_area(
        "批量输入（每行一条标题）",
        height=200,
        placeholder="近期新盘推荐 通州纯新别墅本周开盘\nNBA季后赛 湖人队逆转取胜\n..."
    )

    if st.button("🚀 批量预测", type="primary"):
        # 按换行符分割，去除空行
        titles = [t.strip() for t in batch_text.split("\n") if t.strip()]
        if not titles:
            st.warning("请输入至少一条标题。")
        else:
            results = []
            # st.progress：进度条，初始值 0
            progress = st.progress(0)
            # st.empty：占位符，后续可动态更新内容
            status = st.empty()

            for i, title in enumerate(titles):
                # 动态更新状态文字（显示当前进度）
                status.text(f"正在预测 {i+1}/{len(titles)}：{title[:30]}...")
                result = call_api(title)
                if result:
                    results.append(result)
                # 更新进度条，值为 0.0~1.0
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
                # Counter 统计各类别出现次数
                counter = Counter(r["pred_class"] for r in results)
                chart_df = pd.DataFrame(counter.items(), columns=["类别", "数量"])
                chart_df = chart_df.sort_values("数量", ascending=True)
                # st.bar_chart：内置柱状图，自动渲染
                st.bar_chart(chart_df.set_index("类别"))


# ==========================================================================
# 模式 3：预测历史
# ==========================================================================
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