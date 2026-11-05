"""
====================================================================================================
data_preprocessing.py —— BERT 文本分类数据预处理模块
====================================================================================================
【文件作用】
    1. 加载原始数据文件（txt格式），解析为 (标题, 标签) 元组列表
    2. 自定义 PyTorch Dataset 数据集类
    3. 自定义 collate_fn 批次处理函数，完成分词和向量化
    4. 封装 DataLoader 构建函数，供训练/验证/测试使用

【依赖包详细说明】
----------------------------------------------------------------------------------------------------
1. torch
   PyTorch 深度学习核心库。
   常用 API：
   - torch.tensor(data, dtype=...)    从列表/数组创建张量
       参数：data（数据源）, dtype（数据类型，如 torch.long, torch.float32）
   - torch.cuda.is_available()        检查 GPU 是否可用
   - torch.save(obj, path)            保存模型/张量到文件
   - torch.load(path)                 从文件加载模型/张量
   - torch.no_grad()                  上下文管理器，禁用梯度计算（推理时使用，节省显存）
   - torch.argmax(input, dim)         返回指定维度上最大值的索引

2. transformers（HuggingFace 出品，最流行的 NLP 预训练模型库）
   - BertTokenizer：BERT 分词器，将文本转为 token id 序列
       常用参数：
       - from_pretrained(path/name)    加载预训练分词器，参数可以是本地路径或模型名
       - __call__(texts, ...)          对文本进行分词编码
           参数：
             texts:             待分词文本（str 或 List[str]）
             max_length:        最大序列长度，超过截断，不足填充
             padding:           填充策略
                                True/'max_length'：填充到 max_length
                                False：不填充（默认）
                                'longest'：填充到批次内最长序列
             truncation:        是否截断超长序列（True/False）
             return_tensors:    返回的张量类型
                                'pt'：PyTorch 张量
                                'tf'：TensorFlow 张量
                                None：Python 列表（默认）
             add_special_tokens: 是否添加 [CLS] 和 [SEP] 特殊标记（默认 True）
       返回：BatchEncoding 对象，包含：
             - input_ids:       token 对应的 id 序列
             - attention_mask:  注意力掩码（1=有效token, 0=填充token）
             - token_type_ids:  句子类型标记（单句任务全为0）

   - BertModel：BERT 模型本体，输出隐藏状态
       常用参数：
       - from_pretrained(path/name)    加载预训练模型
       - __call__(input_ids, attention_mask, ...)  前向传播
           返回：BaseModelOutputWithPooling，包含：
             - last_hidden_state:  最后一层隐藏状态 (batch, seq_len, hidden_size=768)
             - pooler_output:      [CLS] 经过线性层+tanh 的输出 (batch, 768)，常用于分类

   - BertConfig：BERT 模型配置
       常用参数：
       - from_pretrained(path/name)    加载预训练配置
       可查看属性：vocab_size, hidden_size, num_hidden_layers, num_attention_heads 等

3. tqdm
   进度条库，用于显示循环进度。
   常用 API：
   - tqdm(iterable, desc="描述", total=总数)
       参数：
         iterable:  可迭代对象
         desc:      进度条前缀描述文字
         total:     总步数（自动推断时可不传）
         ncols:     进度条宽度（字符数），None 为自适应
         unit:      单位描述，如 'samples'

4. torch.utils.data.Dataset
   PyTorch 数据集基类，自定义数据集必须继承它并实现两个方法：
   - __len__(self)    返回数据集样本总数
   - __getitem__(self, index)  返回索引对应的样本

5. torch.utils.data.DataLoader
   PyTorch 数据加载器，封装 Dataset 并提供批量加载、打乱、多进程等功能。
   常用参数：
   - dataset:      数据集实例（必须继承自 Dataset）
   - batch_size:   每批样本数（默认 1）
   - shuffle:      是否每轮打乱数据（默认 False）
   - num_workers:  多进程加载的进程数（0=主进程加载，>0=多进程加速）
   - collate_fn:   自定义批次组合函数，默认会将样本堆叠成张量
   - drop_last:    是否丢弃最后不足 batch_size 的批次（默认 False）
   - pin_memory:   是否将数据锁页到 GPU 内存（使用 GPU 时设为 True 可加速）

【面试高频考点 ⭐】
    1. Dataset 和 DataLoader 的关系？各自职责？
    2. 自定义 Dataset 必须实现哪两个方法？（__len__ 和 __getitem__）
    3. collate_fn 的作用？什么时候需要自定义？（样本不是标准张量时）
    4. BERT 分词器对中文的处理粒度？（字级别，每个汉字一个 token）
    5. [CLS] 和 [SEP] 的 id 分别是多少？（101 和 102）
    6. padding 的三种策略？（max_length、longest、False）
    7. attention_mask 的作用？（标记有效 token，让模型忽略填充位置）
    8. shuffle=True 在训练和评估时的区别？为什么评估时用 False？
    9. num_workers 的作用？设置多少合适？（多进程加载，一般 4~8）
    10. pin_memory 的作用？（锁页内存，加速 CPU→GPU 数据传输）
====================================================================================================
"""
from config import Config
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# ==========================================================================
# 全局变量：加载一次，全局复用，避免重复加载模型
# ==========================================================================
config = Config()

# BertTokenizer.from_pretrained()：
#   加载预训练分词器，包含词表（vocab）和分词规则
#   对于中文 BERT，分词粒度是"字"级别（非词级别），每个汉字为一个 token
bert_tokenizer = BertTokenizer.from_pretrained(config.bert_path)

# BertModel.from_pretrained()：
#   加载预训练 BERT 模型，包含 12 层 Transformer Encoder
#   隐藏层维度 768，多头注意力头数 12
bert_model = BertModel.from_pretrained(config.bert_path)

# BertConfig.from_pretrained()：
#   加载模型配置，可查看 vocab_size、hidden_size、层数等超参数
bert_config = BertConfig.from_pretrained(config.bert_path)


# ==========================================================================
# 函数：load_raw_file
# 作用：加载原始 txt 数据文件，解析为 (标题, 标签) 格式的列表
# ==========================================================================
def load_raw_file(datapath):
    """
    加载并解析原始数据文件。

    数据文件格式（每行）：
        新闻标题\t类别编号
    例："近期新盘推荐 通州纯新别墅本周开盘\t1"

    参数:
        datapath (str): 原始数据文件路径

    返回:
        list[tuple[str, int]]: 处理后的样本列表，格式为 [(标题, 标签), ...]
    """
    # ------------------------------------------------------------------
    # open() 内置函数：
    #   - mode='r':  只读模式
    #   - encoding='utf-8':  指定 UTF-8 编码，保证中文正确读取
    # with 语句：上下文管理器，自动调用 f.close()，即使发生异常也不会泄漏文件句柄
    # ------------------------------------------------------------------
    with open(datapath, mode='r', encoding='utf-8') as f:
        # readlines()：一次性读取所有行，返回 list[str]
        #   注意：大文件用 readlines() 会占大量内存，可改用逐行读取
        lines = f.readlines()

        result_list = []
        # tqdm()：包装可迭代对象，显示进度条，desc 参数设置进度条描述
        for line in tqdm(lines, desc="处理文件中"):
            # str.strip()：去除字符串首尾的空白字符（空格、\n、\r、\t 等）
            line = line.strip()
            if line == '':
                continue

            # str.split(sep)：
            #   按分隔符 sep 切分字符串，返回 list
            #   这里用 '\t'（Tab键）切分，得到 [标题, 标签] 两个元素
            title, label = line.split('\t')

            # str.isdigit()：判断字符串是否全由数字组成
            #   用于过滤异常行（如标签不是数字的情况）
            if not label.isdigit():
                continue

            # int() 将字符串转为整数；tuple() 打包为元组
            result_list.append((title, int(label)))

        return result_list


# ==========================================================================
# 类：NewsDataset
# 作用：自定义 PyTorch 数据集，封装样本列表，提供索引访问接口
# ==========================================================================
class NewsDataset(Dataset):
    """
    自定义数据集类，继承 torch.utils.data.Dataset。

    必须实现的方法：
        __len__()   -> 返回数据集大小
        __getitem__() -> 根据索引返回单个样本

    PyTorch 的 Dataset 是一个抽象类，继承后 DataLoader 才能自动完成
    批量加载、打乱、多进程读取等操作。
    """

    def __init__(self, data_list):
        """
        初始化数据集。

        参数:
            data_list (list[tuple[str, int]]): 样本列表，load_raw_file() 的返回值
        """
        self.data_list = data_list
        self.sample_cnt = len(data_list)

    def __len__(self):
        """
        返回数据集样本总数。

        DataLoader 通过此方法知道总样本数，用于：
        - 计算每个 epoch 的迭代步数
        - 判断是否完成一轮遍历
        """
        return self.sample_cnt

    def __getitem__(self, index):
        """
        根据索引获取单个样本。

        DataLoader 通过此方法逐条获取数据，自动处理：
        - 索引递增遍历
        - shuffle 时随机索引
        - 多进程并行读取（num_workers > 0 时）

        参数:
            index (int): 样本索引，DataLoader 自动传入

        返回:
            tuple[str, int]: (新闻标题, 标签)

        索引安全处理：
            min(max(0, index), self.sample_cnt - 1)
            - max(0, index)：确保索引不小于 0（处理负数索引）
            - min(..., sample_cnt-1)：确保索引不超出范围（否则越界报错）
        """
        index = min(max(0, index), self.sample_cnt - 1)
        title, label = self.data_list[index]
        return title, label


# ==========================================================================
# 函数：collate_fn
# 作用：将一个批次中的原始样本组合成模型可接收的张量格式
# ==========================================================================
def collate_fn(batch_data):
    """
    批次数据组合函数，由 DataLoader 的 collate_fn 参数指定。

    DataLoader 默认的 collate_fn 会将数据堆叠成张量，但这里样本是
    (标题字符串, 标签整数) 的元组，不能直接堆叠，需要自定义处理。

    处理流程：
        1. 将批次中的标题和标签分开
        2. 对标题进行 BERT 分词编码
        3. 将标签转为张量

    参数:
        batch_data (list[tuple[str, int]]): 一个批次的样本列表
            例：[('标题1', 1), ('标题2', 5), ...]

    返回:
        tuple: (input_ids, attention_mask, labels)
            - input_ids (Tensor):      形状 (batch_size, max_length)
            - attention_mask (Tensor): 形状 (batch_size, max_length)
            - labels (Tensor):         形状 (batch_size,)
    """
    # ------------------------------------------------------------------
    # zip(*batch_data) 解包技巧：
    #   zip(*) 将 [(a,1), (b,2), (c,3)] 转为 [(a,b,c), (1,2,3)]
    #   即把"按样本排列"转为"按字段排列"，方便分别处理标题和标签
    # ------------------------------------------------------------------
    titles_tup, labels_tup = zip(*batch_data)

    # ------------------------------------------------------------------
    # bert_tokenizer() 分词编码：
    #   将文本转为 BERT 模型可接受的 token id 序列
    #
    # 参数说明：
    #   texts:             元组/列表形式的文本集合
    #   max_length:        最大序列长度，来自 config
    #                      超过此长度的序列会被截断（truncation=True 时）
    #   padding:           填充到 max_length
    #                      True 或 'max_length'：统一填充到 max_length
    #                      'longest'：填充到批次内最长序列
    #                      False：不填充（默认，仅支持 batch_size=1）
    #   truncation:        是否截断超长序列
    #                      True：截断
    #                      False：不截断（超长会报错）
    #   return_tensors:    返回张量类型
    #                      'pt'：返回 PyTorch Tensor
    #                      'tf'：返回 TensorFlow Tensor
    #                      None：返回 Python list（默认）
    #
    # 返回值 BatchEncoding 对象（类似字典），包含：
    #   .input_ids:        token id 矩阵 (batch_size, max_length)
    #   .attention_mask:   注意力掩码 (batch_size, max_length)
    #                      1 表示有效 token，0 表示填充 token（BERT 忽略填充位置）
    #   .token_type_ids:   句子分割标记（单句分类全为 0）
    # ------------------------------------------------------------------
    data_tensor = bert_tokenizer(
        titles_tup,
        max_length=config.max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    # 解析分词结果
    # input_ids：每个 token 在词表中的索引，形状 (batch_size, seq_len)
    #   [CLS] 的 id 通常是 101，[SEP] 的 id 通常是 102
    input_ids = data_tensor.input_ids
    # attention_mask：1=需要关注的真实 token，0=填充的 padding token
    #   BERT 的自注意力机制会忽略 mask=0 的位置
    attention_mask = data_tensor.attention_mask

    # ------------------------------------------------------------------
    # torch.tensor()：
    #   从 Python 列表/元组创建张量
    #   dtype=torch.long：指定为 64 位整数（分类任务标签常用 long 类型）
    #   常用 dtype：
    #     torch.float32（默认）: 32 位浮点，模型权重常用
    #     torch.long:            64 位整数，标签、索引常用
    #     torch.int32:           32 位整数
    #     torch.bool:            布尔类型，掩码常用
    # ------------------------------------------------------------------
    labels = torch.tensor(labels_tup, dtype=torch.long)

    return input_ids, attention_mask, labels


# ==========================================================================
# 函数：build_dataloader
# 作用：一站式构建 DataLoader，整合加载文件→创建数据集→创建加载器
# ==========================================================================
def build_dataloader(datapath, shuffle=False):
    """
    构建 DataLoader 数据加载器。

    参数:
        datapath (str): 数据文件路径
        shuffle (bool):  是否打乱数据
                         True：训练集使用，每轮随机打乱，防止模型记忆顺序
                         False：验证集/测试集使用，保持顺序便于评估

    返回:
        DataLoader: PyTorch 数据加载器实例
    """
    # 步骤1：加载原始文件，解析为 (标题, 标签) 列表
    data_list = load_raw_file(datapath)

    # 步骤2：创建自定义数据集实例
    # NewsDataset 封装了样本列表，实现了 __len__ 和 __getitem__ 接口
    dataset = NewsDataset(data_list)

    # 步骤3：创建 DataLoader
    # DataLoader 参数说明：
    #   dataset:    数据集实例
    #   batch_size: 每批加载的样本数
    #   shuffle:    是否打乱
    #   collate_fn: 自定义批次组合函数，这里指定为 collate_fn
    #               如果不指定，默认会尝试用 torch.stack 堆叠，但我们的数据
    #               是 (str, int) 元组，无法直接堆叠，所以必须自定义
    data_loader = DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        batch_size=config.batch_size,
        collate_fn=collate_fn
    )

    return data_loader


# ==========================================================================
# 测试入口
# ==========================================================================
if __name__ == '__main__':
    # 测试验证集数据加载
    # build_dataloader(config.dev_datapath, shuffle=False)：
    #   返回一个 DataLoader 迭代器
    #
    # 迭代 DataLoader 时，每次返回一个批次的数据，格式由 collate_fn 决定：
    #   input_ids:      词索引张量 (batch_size, max_length)
    #   attention_mask: 注意力掩码 (batch_size, max_length)
    #   labels:         标签张量 (batch_size,)
    for input_ids, attention_mask, labels in build_dataloader(config.dev_datapath, shuffle=False):
        print("input_ids-->", input_ids)
        print("input_ids 形状:", input_ids.shape)
        print("attention_mask-->", attention_mask)
        print("attention_mask 形状:", attention_mask.shape)
        print("labels-->", labels)
        print("labels 形状:", labels.shape)
        break  # 只打印第一个批次，验证数据加载是否正常