# -*- coding: utf-8 -*-
"""
数据预处理模块
==============
功能：将原始文本数据经过分词、截断等处理后，保存为结构化的 CSV 文件。

处理流程：
    1. 读取原始 txt 文件（\t 分隔的 text 和 label）
    2. 使用 jieba 对每条文本进行中文分词
    3. 截取前 30 个词（控制特征维度，避免过长句子引入噪声）
    4. 保存为 CSV 格式，方便后续模型训练直接加载

依赖：
    - pandas：数据处理
    - jieba：中文分词
    - config.py：项目配置
"""

import pandas as pd
import jieba

from config import Config

# 1- 加载配置文件
config = Config()


def preprocessing(datapath: str) -> None:
    """
    对原始文本数据进行分词预处理并保存。

    参数:
        datapath (str): 原始数据文件路径，文件格式为 \t 分隔的 text 和 label

    知识点详解：
    -----------
    pd.read_csv() 常用参数说明：
        - filepath_or_buffer : str
            文件路径或文件对象。支持本地文件、URL、StringIO 等。
        - sep : str, default ','
            字段分隔符。常用值：','（csv）、'\t'（tsv）、'|' 等。
            如果设为 None，Python 解析器会自动检测分隔符，
            但 C 解析器（默认）不支持自动检测。
        - encoding : str, default None
            文件编码。中文常用 'UTF-8'、'GBK'、'GB2312'。
            如果读取报错 UnicodeDecodeError，通常需要指定正确的编码。
        - names : array-like, optional
            自定义列名列表。当原始文件没有表头时使用，
            如 names=["text", "label"] 会替代默认的列名。
        - header : int or None, default 'infer'
            指定作为列名的行号。None 表示文件没有列名行。
            当指定 names 参数时，header 自动设为 None。
        - nrows : int, optional
            只读取前 n 行，用于预览大文件的前几行。
        - usecols : list-like, optional
            只读取指定的列，可以按列名或列索引选取。

    jieba.lcut() 说明：
        - jieba 是中文分词领域最流行的开源库之一。
        - lcut() 返回分词后的列表（list），与 cut() 返回生成器不同。
        - 还有 lcut_for_search() 适合搜索引擎模式的细粒度分词。
        - 分词模式：
            * 精确模式（默认）：lcut(text, cut_all=False)
            * 全模式：lcut(text, cut_all=True)，把所有可能的词都扫描出来
            * 搜索引擎模式：lcut_for_search(text)，在精确模式基础上再切分长词

    jieba 常用功能：
        - jieba.add_word("新词")：添加自定义词，使分词器能识别新词
        - jieba.del_word("词")：删除词，使分词器忽略某个词
        - jieba.load_userdict("dict.txt")：加载自定义词典文件
        - jieba.suggest_freq(segment, tune=True)：调节单个词语的词频

    df.to_csv() 常用参数说明：
        - path_or_buf : str or file handle
            输出文件路径。设为 None 则返回字符串。
        - sep : str, default ','
            输出字段分隔符。
        - header : bool, default True
            是否写入列名。True 写入，False 不写入。
        - index : bool, default True
            是否写入行索引。通常设为 False 避免冗余列。
        - mode : str, default 'w'
            写入模式。'w' 覆盖写入，'a' 追加写入。
            注意：header 在追加模式下需特殊处理，否则会重复写入表头。
        - encoding : str, default 'utf-8'
            输出文件编码。Windows 下 Excel 打开中文 CSV 可能需要 'gbk' 或 'utf-8-sig'。
        - columns : list, optional
            只输出指定的列。
        - na_rep : str, default ''
            缺失值的替代字符串。
        - float_format : str, optional
            浮点数格式，如 '%.2f' 保留两位小数。
    """
    # 1- 读取文件内容
    # pd.read_csv 虽然名字叫 read_csv，但通过 sep 参数可以读取任意分隔符的文本文件
    df = pd.read_csv(
        datapath,
        encoding="UTF-8",           # 文件编码：UTF-8 是最通用的中文编码
        sep="\t",                   # 分隔符：\t 表示 Tab 分隔（即 TSV 格式）
        names=["text", "label"]     # 列名：原始文件无表头，手动指定列名
    )

    # 2- 对新闻标题进行分词，然后进行句子长度的处理只保留前 30 个词
    # line 代表每行中的 text 内容，即新闻标题
    #
    # 知识点：lambda 表达式
    #     lambda 是 Python 的匿名函数，语法为 lambda 参数: 表达式
    #     等价于：
    #         def func(line):
    #             return ' '.join(jieba.lcut(line)[:30])
    #
    # 知识点：' '.join() 的作用
    #     将分词后的列表用空格拼接成一个字符串。
    #     例如：['苹果', '非常', '好吃'] -> '苹果 非常 好吃'
    #     这样做的目的是让 TfidfVectorizer 能按空格识别每个词。
    #
    # 知识点：[:30] 切片截断
    #     只保留前 30 个词，原因：
    #     1. 控制特征维度：词越多，TF-IDF 特征维度越大，训练越慢
    #     2. 新闻标题通常较短，30 个词已足够覆盖关键信息
    #     3. 过长句子可能包含噪声，截断有正则化效果
    df['words'] = df['text'].apply(
        lambda line: ' '.join(jieba.lcut(line)[:30])
    )

    # 3- 保存处理后的文件
    # 根据输入路径判断数据集类型，保存到对应的预处理文件路径
    if datapath == config.train_datapath:
        df.to_csv(
            config.process_train_datapath,
            sep=",",                 # 输出为逗号分隔的标准 CSV 格式
            header=True,             # 写入列名行
            index=False,             # 不写入行索引（避免多余列）
            mode="w",                # 覆盖写入模式
            encoding="UTF-8"
        )
        print("train数据预处理完成，保存成功")

    elif datapath == config.test_datapath:
        df.to_csv(
            config.process_test_datapath,
            sep=",",
            header=True,
            index=False,
            mode="w",
            encoding="UTF-8"
        )
        print("test数据预处理完成，保存成功")

    elif datapath == config.dev_datapath:
        df.to_csv(
            config.process_dev_datapath,
            sep=",",
            header=True,
            index=False,
            mode="w",
            encoding="UTF-8"
        )
        print("dev数据预处理完成，保存成功")


if __name__ == '__main__':
    # 2- 依次对训练集、测试集、验证集进行预处理
    for datapath in [config.train_datapath, config.test_datapath, config.dev_datapath]:
        preprocessing(datapath)