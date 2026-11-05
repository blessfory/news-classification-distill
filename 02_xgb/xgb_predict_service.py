# -*- coding: utf-8 -*-
"""
XGBoost 预测服务核心逻辑
========================

负责加载 Word2Vec 词向量模型和 XGBoost 分类器，对外提供单条预测接口。

预测流程：
    1. jieba 分词 → 截取前 30 个词
    2. Word2Vec 查词向量 → 取平均 → 128 维密集向量
    3. XGBoost 预测 → 类别 ID
    4. 类别 ID → 类别名称（通过 id_class_dict 映射）

与 01_rf 的 rf_predict_service.py 的区别：
    - 用 Word2Vec 替代 TF-IDF 做特征提取（稀疏 → 密集）
    - 用 XGBoost 替代 RandomForest 做分类（Bagging → Boosting）
"""

import jieba
import joblib
import numpy as np
from gensim.models import Word2Vec

from config import Config

# ============================================================
# 1- 加载配置
# ============================================================
config = Config()

# ============================================================
# 2- 加载训练好的模型（启动时加载一次，避免每次请求重复 IO）
# ============================================================

# Word2Vec 用 gensim 原生格式加载（非 joblib）
# gensim 的 .save() / .load() 是专用的二进制格式，不适合用 joblib
w2v_model = Word2Vec.load(config.w2v_model_path)
print(f"[INFO] Word2Vec 模型已加载，词汇表大小：{len(w2v_model.wv)}")

# XGBoost 用 joblib 加载（与 sklearn 生态兼容）
xgb_model = joblib.load(config.xgb_model_path)
print(f"[INFO] XGBoost 模型已加载")

# ============================================================
# 3- 预加载类别映射字典（启动时读取一次，避免每次预测重复 IO）
# ============================================================
# 字典格式：{0: 'finance', 1: 'realty', ...}
with open(config.class_datapath, mode="r", encoding="UTF-8") as f:
    id_class_dict = {i: class_name.strip() for i, class_name in enumerate(f)}
print(f"[INFO] 已加载类别映射：{id_class_dict}")

# ============================================================
# 4- 工具函数：文本 → Word2Vec 平均向量
# ============================================================
def text_to_vector(words_text: str) -> np.ndarray:
    """
    将分词后的文本转为 Word2Vec 平均向量

    算法：
        1. 按空格拆词
        2. 查 Word2Vec 获取每个词的向量
        3. 取平均（未命中词跳过，全未命中返回零向量）

    参数：
        words_text : str — 空格分隔的分词结果，如 "今天 天气 真的 很好"

    返回：
        np.ndarray — 形状 (128,)，即 128 维密集向量
    """
    words = words_text.split()
    vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]

    if len(vectors) == 0:
        return np.zeros(128)

    return np.mean(vectors, axis=0)


# ============================================================
# 5- 预测函数
# ============================================================
def predict(news_data: dict) -> dict:
    """
    对单条新闻标题进行预测

    参数:
        news_data : dict — {"title": "新闻标题内容"}

    返回:
        dict — {"title": "新闻标题内容", "pred_class": "预测的新闻类别"}
    """
    # 1- 取出标题；jieba 分词；截取前 30 个词；空格连接
    words = " ".join(jieba.lcut(news_data["title"])[:30])

    # 2- Word2Vec 向量化：词列表 → 128 维密集向量
    vec = text_to_vector(words)

    # 3- XGBoost 预测，取第一条结果
    #    XGBoost 的 predict 返回 array，[0] 取第一个元素
    y_pred_index = int(xgb_model.predict([vec])[0])

    # 4- 类别 ID → 类别名称
    pred_class_name = id_class_dict[y_pred_index]

    news_data["pred_class"] = pred_class_name
    return news_data


# ============================================================
# 6- 本地测试
# ============================================================
def predict_unified(text: str) -> dict:
    """
    统一预测接口：输入文本字符串，返回统一格式。

    参数:
        text : str — 新闻标题文本

    返回:
        dict — {"label_id": int, "label": str, "confidence": float}
    """
    words = " ".join(jieba.lcut(text)[:30])
    vec = text_to_vector(words)
    y_pred_index = int(xgb_model.predict([vec])[0])
    proba = float(xgb_model.predict_proba([vec])[0][y_pred_index])
    pred_class_name = id_class_dict[y_pred_index]

    return {
        "label_id": y_pred_index,
        "label": pred_class_name,
        "confidence": round(proba, 4)
    }


if __name__ == '__main__':
    news_data = {"title": "同步A股首秀：港股缩量回调"}
    result = predict(news_data)
    print(result)

    result2 = predict_unified("同步A股首秀：港股缩量回调")
    print(result2)