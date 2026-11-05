"""
====================================================================================================
bert_model.py —— BERT 文本分类模型定义
====================================================================================================
【文件作用】
    定义 BERT 文本分类模型，使用"预训练 BERT + 自定义全连接层"的架构。
    采用"特征提取"（Feature Extraction）模式：冻结 BERT 参数，只训练分类头。

【依赖包详细说明】
----------------------------------------------------------------------------------------------------
1. torch
   PyTorch 深度学习核心库。
   常用 API：
   - torch.tensor(data, dtype=...)     从列表/数组创建张量
   - torch.cuda.is_available()         检查 GPU 是否可用
   - torch.save(obj, path)             保存模型/张量
   - torch.load(path)                  加载模型/张量
   - torch.no_grad()                   上下文管理器，禁用梯度计算
   - torch.argmax(input, dim)          返回指定维度上最大值的索引
   - torch.max(input, dim)             返回最大值及其索引
   - torch.cat(tensors, dim)           沿指定维度拼接张量
   - torch.stack(tensors, dim)         沿新维度堆叠张量

2. torch.nn（神经网络模块）
   PyTorch 构建神经网络的核心模块。
   常用 API：
   - nn.Module:                        所有神经网络模型的基类，必须继承
       关键方法：
       - __init__():  定义网络层（各个子模块）
       - forward(x):  定义前向传播逻辑
       - parameters(): 返回模型所有可训练参数
       - named_parameters(): 返回参数名称和参数值
       - train():     设置为训练模式（启用 Dropout、BatchNorm 等）
       - eval():      设置为评估模式（禁用 Dropout、BatchNorm 等）
       - to(device):  将模型迁移到指定设备（CPU/GPU）
       - save/load:   通过 torch.save/load 保存和加载

   - nn.Linear(in_features, out_features, bias=True):  全连接层（线性变换层）
       参数：
         in_features:  输入特征维度
         out_features: 输出特征维度
         bias:         是否使用偏置项（默认 True）
       数学公式：y = x @ W^T + b
       形状变化：(batch, in_features) -> (batch, out_features)

   - nn.Dropout(p=0.5):               Dropout 层，随机丢弃神经元，防止过拟合
       参数：p=丢弃概率（默认 0.5）

   - nn.CrossEntropyLoss():           交叉熵损失，自动包含 softmax
       参数：
         weight:    类别权重，用于处理类别不平衡
         reduction: 'mean'（默认，返回均值）, 'sum'（求和）, 'none'（不聚合）

   - nn.functional 下的函数式 API：
     - F.relu(x):                     ReLU 激活函数
     - F.softmax(x, dim):             Softmax，沿指定维度归一化
     - F.log_softmax(x, dim):         LogSoftmax，数值更稳定
     - F.dropout(x, p, training):     Dropout 函数式版本

3. transformers（HuggingFace 预训练模型库）
   - BertModel：BERT 预训练模型
       输入：input_ids (batch, seq_len), attention_mask (batch, seq_len)
       输出：BaseModelOutputWithPooling 对象
         - last_hidden_state:  (batch, seq_len, 768) 最后一层所有 token 的隐藏状态
         - pooler_output:      (batch, 768) [CLS] 经过线性层+tanh 的输出

   - BertConfig：BERT 模型配置
       常用属性：
         - hidden_size:        隐藏层维度（BERT-base 为 768）
         - num_hidden_layers:  Transformer 层数（BERT-base 为 12）
         - num_attention_heads: 注意力头数（BERT-base 为 12）
         - vocab_size:         词表大小（BERT-base-chinese 约 21128）
         - max_position_embeddings: 最大位置编码长度（512）

【两种使用 BERT 的范式】
   1. 特征提取（Feature Extraction）—— 本文件采用的方式
      - 冻结 BERT 参数（requires_grad=False）
      - BERT 只作为特征提取器，不参与训练
      - 优点：训练快，显存占用少，适合小数据集
      - 缺点：效果可能不如微调

   2. 微调（Fine-tuning）
      - BERT 参数参与训练（requires_grad=True，默认）
      - 整个模型端到端训练
      - 优点：效果好，能适应特定任务
      - 缺点：训练慢，显存占用大，需要更多数据

【面试高频考点 ⭐】
    1. BERT 的两种使用方式？特征提取 vs 微调的区别和适用场景？
    2. 为什么要冻结 BERT 参数？（节省显存、加速训练、防止小数据集过拟合）
    3. nn.Module 的 forward() 方法如何被调用？（model(x) 自动调用，不直接调 forward）
    4. torch.no_grad() 的作用？为什么推理时必须使用？
    5. [CLS] token 的作用？为什么用它做分类？
    6. BERT-base 的参数量？（约 110M）隐藏层维度？（768）层数？（12）
    7. BERT 输入包含哪些？（input_ids、attention_mask、token_type_ids）
    8. attention_mask 的作用？（让 BERT 忽略填充的 [PAD] token）
    9. nn.Linear 的权重形状？（out_features, in_features）
    10. CrossEntropyLoss 内部做了什么？（自动做 softmax + NLLLoss）
    11. Dropout 的原理？训练和推理时的区别？
    12. 过拟合的常见解决办法？（Dropout、正则化、早停、数据增强）
====================================================================================================
"""
import torch
import torch.nn as nn
from config import Config
from transformers import BertConfig, BertModel

# ==========================================================================
# 全局变量
# ==========================================================================
config = Config()


# ==========================================================================
# 类：BertClassifierModel
# 作用：BERT 文本分类模型（特征提取模式）
# ==========================================================================
class BertClassifierModel(nn.Module):
    """
    BERT 文本分类模型。

    架构：
        BERT（冻结参数）→ 取 [CLS] 向量 → 全连接层 → 分类 logits

    继承关系：
        nn.Module 是所有神经网络模型的基类，继承后需要实现：
        - __init__():  定义网络层
        - forward():   定义前向传播计算过程

    模型结构示意：
        输入: (input_ids, attention_mask)
          │
          ▼
        BERT（参数冻结，不训练）
          │
          ▼
        last_hidden_state[:, 0]  → 取 [CLS] token 的向量 (batch, 768)
          │
          ▼
        nn.Linear(768, class_cnt)  → 全连接分类头
          │
          ▼
        输出: logits (batch, class_cnt)
    """

    def __init__(self):
        """
        初始化模型：定义网络的各个层。

        执行顺序：
            1. super().__init__()：调用 nn.Module 的构造函数，注册模型参数追踪机制
            2. 加载预训练 BERT 模型
            3. 冻结 BERT 参数（特征提取模式）
            4. 定义分类头（全连接层）
        """
        # ------------------------------------------------------------------
        # super().__init__()：
        #   必须调用父类 nn.Module 的构造函数
        #   作用：初始化 PyTorch 内部的参数注册机制（_parameters, _modules 等）
        #   不调用会导致模型参数无法被正确追踪和管理
        # ------------------------------------------------------------------
        super().__init__()

        # ==================================================================
        # 2.1- 加载预训练 BERT 模型
        #
        # BertModel.from_pretrained(config.bert_path)：
        #   加载预训练 BERT 模型，包含：
        #   - Embedding 层：token embedding + position embedding + segment embedding
        #   - 12 层 Transformer Encoder（BERT-base）
        #   - Pooler 层：[CLS] 向量 → Linear → Tanh
        #
        # 模型大小：BERT-base 约 110M 参数
        # ==================================================================
        self.bert_model = BertModel.from_pretrained(config.bert_path)

        # ==================================================================
        # 2.2- 冻结 BERT 参数（特征提取模式）
        #
        # param.requires_grad_(False)：
        #   - requires_grad 是每个张量的属性，决定是否需要计算梯度
        #   - 设为 False 后，反向传播时跳过该参数，不更新
        #   - requires_grad_(False)：原地修改（in-place），末尾的 _ 表示原地操作
        #
        # self.bert_model.parameters()：
        #   - 返回模型所有可训练参数的迭代器
        #   - 每个参数是 torch.nn.Parameter 类型（Tensor 的子类）
        #
        # 冻结与不冻结的对比：
        #   ┌──────────┬─────────────────────┬──────────────────────┐
        #   │          │  冻结 (False)        │  不冻结 (True)       │
        #   ├──────────┼─────────────────────┼──────────────────────┤
        #   │ 训练速度  │  快                  │  慢                  │
        #   │ 显存占用  │  少                  │  多                  │
        #   │ 模型效果  │  一般                │  好（通常）           │
        #   │ 适用场景  │  小数据集、快速实验    │  大数据集、追求效果    │
        #   └──────────┴─────────────────────┴──────────────────────┘
        # ==================================================================
        for param in self.bert_model.parameters():
            param.requires_grad_(False)

        # ==================================================================
        # 2.3- 定义分类头（全连接层）
        #
        # BertConfig.from_pretrained(path).hidden_size：
        #   获取 BERT 隐藏层维度，BERT-base 为 768
        #
        # nn.Linear(in_features, out_features)：
        #   全连接层，执行线性变换 y = x @ W^T + b
        #   - in_features=768：BERT [CLS] 向量的维度
        #   - out_features=class_cnt：类别数量
        #   - 权重 W 形状：(class_cnt, 768)，偏置 b 形状：(class_cnt,)
        #
        # 输入输出形状：
        #   输入:  (batch_size, 768)   → [CLS] 向量
        #   输出:  (batch_size, class_cnt) → 每个类别的 logits（未归一化分数）
        # ==================================================================
        bert_hidden_size = BertConfig.from_pretrained(config.bert_path).hidden_size
        self.linear = nn.Linear(bert_hidden_size, config.class_cnt)

    def forward(self, input_ids, attention_mask):
        """
        前向传播：定义数据从输入到输出的计算流程。

        PyTorch 自动调用此方法（通过 model(input_ids, mask) 触发），
        不需要手动调用 forward()。

        参数:
            input_ids (Tensor):       词索引张量，形状 (batch_size, max_length)
                                      例：[[101, 123, 456, ..., 102, 0, 0]]
                                      [CLS]=101, [SEP]=102, [PAD]=0
            attention_mask (Tensor):  注意力掩码，形状 (batch_size, max_length)
                                      1=有效token需关注，0=填充token需忽略

        返回:
            Tensor: 分类 logits，形状 (batch_size, class_cnt)
                    未经过 softmax，直接输入 CrossEntropyLoss 即可
        """
        # ==================================================================
        # 步骤1：BERT 特征提取
        #
        # torch.no_grad() 上下文管理器：
        #   作用：禁用梯度计算，不构建计算图
        #   效果：节省显存，加速前向传播
        #   原理：PyTorch 默认会追踪所有张量操作，构建计算图用于反向传播。
        #        在 no_grad() 中，所有操作都不会被记录，无法反向传播。
        #
        # 使用场景：
        #   - 推理/预测阶段（必须使用）
        #   - 特征提取模式下的 BERT 前向传播（本文件场景）
        #   - 计算评估指标时
        #
        # 注意：如果 BERT 参数已冻结（requires_grad=False），
        #       即使不加 no_grad()，参数也不会更新，但 PyTorch 仍会
        #       构建计算图（浪费显存），所以加上 no_grad() 更高效。
        # ==================================================================
        with torch.no_grad():
            # self.bert_model(input_ids, attention_mask)：
            #   调用 BERT 模型的前向传播，返回 BaseModelOutputWithPooling 对象
            bert_output = self.bert_model(input_ids, attention_mask)

        # ==================================================================
        # 步骤2：提取 [CLS] 向量 + 分类
        #
        # bert_output.last_hidden_state：
        #   形状 (batch_size, seq_len, 768)
        #   最后一层 Transformer Encoder 的输出，包含每个 token 的隐藏状态
        #
        # bert_output.last_hidden_state[:, 0]：
        #   [:, 0] 表示取所有 batch 的第 0 个 token，即 [CLS] token
        #   BERT 在每个序列开头添加 [CLS] 标记，其向量可代表整个序列的语义
        #   形状：(batch_size, 768)
        #
        # [CLS] vs pooler_output 的区别：
        #   ┌──────────────────────┬──────────────────────────────────────┐
        #   │ last_hidden[:, 0]     │ 最后一层 Transformer 的 [CLS] 原始输出 │
        #   │                      │ 未经过额外变换                         │
        #   ├──────────────────────┼──────────────────────────────────────┤
        #   │ pooler_output        │ [CLS] 向量 → Linear(768,768) → Tanh  │
        #   │                      │ 多了一层线性变换和激活函数               │
        #   └──────────────────────┴──────────────────────────────────────┘
        #   推荐：分类任务使用 last_hidden_state[:, 0]，因为：
        #       1. 保留了原始语义信息，没有经过额外的变换
        #       2. 很多论文和实践中效果更好
        #       3. pooler_output 是为 NSP（下一句预测）任务设计的
        # ==================================================================
        return self.linear(bert_output.last_hidden_state[:, 0])


# ==========================================================================
# 测试入口：验证模型结构是否正确
# ==========================================================================
if __name__ == '__main__':
    # 实例化模型
    model = BertClassifierModel()

    # 打印模型结构
    # print(model) 会显示所有子模块的名称和结构
    print("=" * 60)
    print("模型结构：")
    print(model)
    print("=" * 60)

    # 统计模型参数
    # model.parameters() 返回所有参数迭代器
    # param.requires_grad 标识参数是否需要训练
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量:       {total_params:,}")
    print(f"可训练参数量:   {trainable_params:,}")
    print(f"冻结参数量:     {total_params - trainable_params:,}")
    print("=" * 60)

    # 测试前向传播：构造一个假批次数据
    # 模拟 batch_size=2, max_length=10 的输入
    dummy_input_ids = torch.randint(100, 20000, (2, 10))
    dummy_attention_mask = torch.ones(2, 10, dtype=torch.long)

    # 前向传播
    output = model(dummy_input_ids, dummy_attention_mask)

    # 输出形状应为 (2, class_cnt)，即 (batch_size, 类别数)
    print(f"输入形状:   input_ids={dummy_input_ids.shape}")
    print(f"输出形状:   {output.shape}")
    print(f"输出 logits: {output}")
    print("=" * 60)