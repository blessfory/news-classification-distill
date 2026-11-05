# -*- coding: utf-8 -*-
"""
数据预处理 + 向量化模块
========================
对原始文本数据做 jieba 分词、Word2Vec 训练、向量化，保存为 CSV 和 numpy 数组。

处理流程：
    1. 读取原始 txt 文件（\t 分隔的 text 和 label）
    2. jieba 中文分词 → 截取前 30 个词 → 保存为 CSV
    3. 在训练集上训练 Word2Vec 词向量模型
    4. 将三份数据转为 128 维向量 → 保存为 .npy 文件

设计理念：
    把特征工程（分词 + 向量化）全部放在预处理阶段完成，
    训练脚本和推理服务直接读取处理好的 numpy 数组，职责更清晰。
"""

import pandas as pd
import numpy as np
import jieba
from gensim.models import Word2Vec
from config import Config

config = Config()


# ============================================================
# 阶段一：分词
# ============================================================
def preprocessing(datapath: str) -> None:
    """
    对原始文本进行 jieba 分词预处理并保存为 CSV。

    参数:
        datapath : str — 原始数据文件路径（\t 分隔的 txt）
    """
    # 1- 读取原始文件
    df = pd.read_csv(
        datapath,
        encoding="UTF-8",
        sep="\t",
        names=["text", "label"],
    )

    # 2- jieba 分词 + 截取前 30 个词
    df['words'] = df['text'].apply(
        lambda line: ' '.join(jieba.lcut(line)[:30])
    )

    # 3- 保存
    if datapath == config.train_datapath:
        df.to_csv(config.process_train_datapath, sep=",", header=True, index=False, mode="w", encoding="UTF-8")
        print("train 分词完成")
    elif datapath == config.test_datapath:
        df.to_csv(config.process_test_datapath, sep=",", header=True, index=False, mode="w", encoding="UTF-8")
        print("test 分词完成")
    elif datapath == config.dev_datapath:
        df.to_csv(config.process_dev_datapath, sep=",", header=True, index=False, mode="w", encoding="UTF-8")
        print("dev 分词完成")


# ============================================================
# 阶段二：文本 → Word2Vec 平均向量
# ============================================================
def text_to_w2v_vector(words_text: str, w2v_model: Word2Vec, dim: int = 128) -> np.ndarray:
    """
    将空格分隔的分词文本转为 Word2Vec 平均向量。

    算法：
        1. 按空格拆词
        2. 查 Word2Vec 获取每个词的向量
        3. 取平均（未命中词跳过，全未命中返回零向量）

    参数：
        words_text : str — 空格分隔的分词结果，如 "今天 天气 真的 很好"
        w2v_model  : Word2Vec — 已训练的词向量模型
        dim        : int — 向量维度，默认 128

    返回：
        np.ndarray — 形状 (dim,)，即 dim 维密集向量
    """
    words = words_text.split()
    vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if len(vectors) == 0:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)


# ============================================================
# 阶段三：向量化（分词 → Word2Vec → numpy 保存）
# ============================================================
def vectorize_and_save():
    """
    在训练集上训练 Word2Vec，然后将三份数据全部向量化并保存为 .npy 文件。
    """
    print("\n" + "=" * 50)
    print("向量化阶段：训练 Word2Vec + 转密集向量")
    print("=" * 50)

    # ---- 3.1 读取分词后的数据 ----
    df_train = pd.read_csv(config.process_train_datapath)
    df_dev   = pd.read_csv(config.process_dev_datapath)
    df_test  = pd.read_csv(config.process_test_datapath)
    print(f"训练集：{len(df_train)} 条")
    print(f"验证集：{len(df_dev)} 条")
    print(f"测试集：{len(df_test)} 条")

    # ---- 3.2 在训练集上训练 Word2Vec ----
    sentences = [text.split() for text in df_train["words"]]
    print(f"\n语料库句子数：{len(sentences)}")
    print("开始训练 Word2Vec...")

    w2v_model = Word2Vec(
        sentences,
        vector_size=128,    # 向量维度
        window=5,           # 上下文窗口大小
        min_count=1,        # 最小词频
        workers=8,          # 并行线程数
        seed=88,            # 随机种子
    )
    print(f"Word2Vec 训练完成，词汇表大小：{len(w2v_model.wv)}")

    # ---- 3.3 将三份数据转为 128 维向量 ----
    print("\n将文本转为 Word2Vec 向量...")
    X_train = np.array([text_to_w2v_vector(text, w2v_model) for text in df_train["words"]])
    y_train = df_train["label"].values

    X_dev = np.array([text_to_w2v_vector(text, w2v_model) for text in df_dev["words"]])
    y_dev = df_dev["label"].values

    X_test = np.array([text_to_w2v_vector(text, w2v_model) for text in df_test["words"]])
    y_test = df_test["label"].values

    print(f"训练集特征形状：{X_train.shape}")
    print(f"验证集特征形状：{X_dev.shape}")
    print(f"测试集特征形状：{X_test.shape}")

    # ---- 3.4 保存为 numpy 数组 ----
    np.save(config.vector_train_datapath, X_train)
    np.save(config.vector_train_label_datapath, y_train)
    np.save(config.vector_dev_datapath, X_dev)
    np.save(config.vector_dev_label_datapath, y_dev)
    np.save(config.vector_test_datapath, X_test)
    np.save(config.vector_test_label_datapath, y_test)
    print("向量化数据已保存到 data/ 目录")

    # ---- 3.5 保存 Word2Vec 模型（供推理服务使用） ----
    w2v_model.save(config.w2v_model_path)
    print(f"Word2Vec 模型已保存到：{config.w2v_model_path}")


# ============================================================
# 主程序入口
# ============================================================
if __name__ == '__main__':
    # 阶段一：分词
    print("=" * 50)
    print("阶段一：jieba 分词")
    print("=" * 50)
    for datapath in [config.train_datapath, config.test_datapath, config.dev_datapath]:
        preprocessing(datapath)

    # 阶段二 + 三：向量化
    vectorize_and_save()

    print("\n全部预处理完成！可以运行 xgb_train.py 开始训练了。")