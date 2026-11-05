"""
====================================================================================================
student_bilstm_model.py —— 知识蒸馏：学生模型（BiLSTM）定义
====================================================================================================
【文件作用】
    定义知识蒸馏中的"学生模型"（Student Model）—— 双向 LSTM 分类模型。
    学生模型参数量远小于教师模型（BERT ~110M），通过蒸馏学习教师模型的软标签，
    在保持较好效果的同时，大幅提升推理速度。

【BiLSTM 核心知识点】
----------------------------------------------------------------------------------------------------
1. 为什么选择 BiLSTM 作为学生模型？
   - 参数量小：本模型约 5.5M 参数，是 BERT（110M）的 1/20
   - 推理速度快：不需要 12 层 Transformer 自注意力计算
   - 序列建模能力强：LSTM 天然适合文本序列，双向能捕获上下文信息
   - 部署友好：不需要 HuggingFace 等重型依赖，纯 PyTorch 即可

2. LSTM（长短期记忆网络）核心原理
   LSTM 通过三个"门"控制信息流动，解决普通 RNN 的梯度消失问题：

   ┌─────────────┬──────────────────────────────────────────────────────┐
   │ 遗忘门        │ f_t = σ(W_f·[h_{t-1}, x_t] + b_f)                  │
   │ Forget Gate  │ 决定丢弃哪些旧信息（0=全忘，1=全保留）                 │
   ├─────────────┼──────────────────────────────────────────────────────┤
   │ 输入门        │ i_t = σ(W_i·[h_{t-1}, x_t] + b_i)                  │
   │ Input Gate   │ 决定哪些新信息存入细胞状态                            │
   ├─────────────┼──────────────────────────────────────────────────────┤
   │ 输出门        │ o_t = σ(W_o·[h_{t-1}, x_t] + b_o)                  │
   │ Output Gate  │ 决定当前时刻输出什么                                  │
   ├─────────────┼──────────────────────────────────────────────────────┤
   │ 候选细胞状态   │ C̃_t = tanh(W_c·[h_{t-1}, x_t] + b_c)              │
   │ 细胞状态更新   │ C_t = f_t * C_{t-1} + i_t * C̃_t                   │
   │ 隐藏状态输出   │ h_t = o_t * tanh(C_t)                             │
   └─────────────┴──────────────────────────────────────────────────────┘

3. 双向 LSTM（BiLSTM）
   - 单向 LSTM 只能看到"上文"信息（从左到右）
   - 双向 LSTM 同时处理正向和反向序列，拼接后同时看到"上文+下文"
   - 输出维度 = hidden_size × 2（两个方向拼接）

   例：句子"今天天气很好"
      正向 LSTM：今天 → 天气 → 很好  （看到上文）
      反向 LSTM：很好 → 天气 → 今天  （看到下文）
      拼接后同时包含上下文信息

4. 词嵌入层（Embedding Layer）
   - 将离散的 token id 映射为稠密向量
   - 本质是一个查找表：vocab_size × embedding_dim 的矩阵
   - 本模型使用 BERT 的词表（vocab_size ≈ 21128），但 embedding_dim 只有 256
   - 训练过程中词向量会不断更新，学习到语义信息

5. 平均池化（Average Pooling）vs 其他池化方式
   ┌──────────────────┬──────────────────────────────────────────────────┐
   │ 平均池化（本模型）  │ 所有有效词的向量求平均，平滑稳定                     │
   │                  │ 适合：句子整体语义分类                             │
   ├──────────────────┼──────────────────────────────────────────────────┤
   │ 最大池化          │ 取每个维度最大值，保留最显著特征                     │
   │                  │ 适合：关键词匹配型任务                             │
   ├──────────────────┼──────────────────────────────────────────────────┤
   │ [CLS] 向量        │ 取第一个 token 的向量（BERT 做法）                 │
   │                  │ 需要额外训练一个"汇总"能力                         │
   └──────────────────┴──────────────────────────────────────────────────┘

6. 模型参数量估算
   - Embedding:  21128 × 256 = 5,408,768
   - BiLSTM(3层): 约 4 × 256 × 256 × 2 × 3 ≈ 1,572,864（每层4个门×输入×隐藏×双向×层数）
   - Linear:     512 × N_class ≈ 几千
   - 总计约 9.6M 参数（其中 embedding 5.41M），约为 BERT 110M 的 1/11

7. 学生模型训练方式对比
   ┌──────────────────────┬──────────────────────────────────────────────┐
   │ 仅用硬标签训练         │ 直接用 CrossEntropyLoss + 真实标签            │
   │                      │ 效果一般，小模型容量有限，学不到暗知识            │
   ├──────────────────────┼──────────────────────────────────────────────┤
   │ 蒸馏训练（本方案）     │ 硬标签 Loss + 软标签 Loss（教师模型输出）       │
   │                      │ 效果更好，学生学到了教师"认为哪些类别容易混淆"     │
   └──────────────────────┴──────────────────────────────────────────────┘

【依赖包说明】
----------------------------------------------------------------------------------------------------
1. torch
   PyTorch 深度学习核心库。
   - torch.tensor(data):            创建张量
   - tensor.unsqueeze(dim):         在指定维度增加一个维度
   - tensor.sum(dim):               沿指定维度求和
   - tensor.bool():                 转为布尔类型

2. torch.nn
   - nn.Module:                     所有神经网络模型的基类
   - nn.Embedding(num_embeddings, embedding_dim):  词嵌入层
       参数：
         num_embeddings:  词表大小（vocab_size）
         embedding_dim:   每个词的向量维度
       输入：(batch, seq_len) 的整数索引
       输出：(batch, seq_len, embedding_dim) 的浮点向量
   - nn.LSTM(input_size, hidden_size, num_layers, batch_first, bidirectional):
       多层双向 LSTM
       参数：
         input_size:    输入特征维度（= embedding_dim）
         hidden_size:   隐藏层维度
         num_layers:    LSTM 层数（堆叠层数）
         batch_first:   True 时输入形状为 (batch, seq, feature)
         bidirectional: True 时使用双向 LSTM
       输出：(output, (h_n, c_n))
         - output: (batch, seq_len, hidden_size × 2)  所有时间步的隐藏状态
         - h_n:    (num_layers × 2, batch, hidden_size)  最后一层隐藏状态
         - c_n:    (num_layers × 2, batch, hidden_size)  最后一层细胞状态
   - nn.Dropout(p):                 随机失活层，p=丢弃概率
   - nn.Linear(in_features, out_features):  全连接层

3. transformers
   - BertConfig:  BERT 配置，用于获取 vocab_size（复用 BERT 词表）

【面试高频考点 ⭐】
    1. LSTM 的三个门分别是什么？各自的作用？
    2. BiLSTM 和单向 LSTM 的区别？为什么双向效果更好？
    3. 为什么学生模型用 BiLSTM 而不是 BERT？（参数量、推理速度、部署成本）
    4. Embedding 层的原理？输入输出形状？
    5. 平均池化和最大池化的区别？各自适用场景？
    6. LSTM 如何解决 RNN 的梯度消失问题？
    7. batch_first=True 的作用？
    8. 为什么 sum(dim=1) 能得到句子级别的向量？
    9. 为什么池化时要除以有效词数量（均值）而不是直接求和？
    10. 知识蒸馏中，学生模型的结构可以和教师模型不同吗？为什么？
    11. 蒸馏后的 BiLSTM 效果能接近 BERT 吗？差距大概多少？
    12. 如果蒸馏效果不好，可能的原因有哪些？（温度T不合适、α权重不对、学生模型太小）
====================================================================================================
"""
import torch
import torch.nn as nn
from config import Config
from transformers import BertConfig

# ==========================================================================
# 全局变量
# ==========================================================================
config = Config()

# 加载 BERT 配置，复用 BERT 的词表大小（vocab_size ≈ 21128）
# 这样做的好处：学生模型和教师模型共享同一套 token id，分词结果一致
bert_config = BertConfig.from_pretrained(config.bert_path)


# ==========================================================================
# 类：BiLSTMStuModel
# 作用：知识蒸馏的学生模型（BiLSTM 分类模型）
#
# ==========================================================================
# 【模型蒸馏知识点：学生模型的设计原则】
#   1. 结构灵活：学生模型结构可以和教师完全不同（教师是 Transformer，学生是 LSTM）
#   2. 参数量小：学生参数量通常为教师的 1/10 ~ 1/100
#   3. 输出对齐：学生输出的 logits 维度必须和教师一致（都是 class_cnt 维）
#   4. 共享词表：学生复用教师的 tokenizer/词表，保证输入一致
# ==========================================================================
class BiLSTMStuModel(nn.Module):
    """
    知识蒸馏学生模型：Embedding + BiLSTM + 平均池化 + 全连接分类头。

    架构：
        input_ids → Embedding → 掩码过滤 → BiLSTM → 平均池化 → Dropout → Linear → logits

    参数量：约 9.6M（BERT 的 1/11）

    继承关系：
        nn.Module 是所有神经网络模型的基类，必须实现：
        - __init__():  定义网络层
        - forward():   定义前向传播
    """

    def __init__(self):
        """
        初始化学生模型：定义 Embedding + BiLSTM + Dropout + Linear。

        关键步骤：
            1. super().__init__()：初始化 nn.Module 的参数追踪
            2. 设置超参数：词表大小、词向量维度、隐藏层维度、LSTM 层数
            3. 搭建网络：词嵌入层 → BiLSTM → Dropout → 全连接层
        """
        # ==================================================================
        # 1- 初始化父类
        #
        # super().__init__()：
        #   调用 nn.Module 的构造函数，初始化参数管理机制
        #   必须调用，否则 model.parameters() 等方法无法正常工作
        # ==================================================================
        super().__init__()

        # ==================================================================
        # 2- 设置超参数
        #
        # vocab_size：词表大小，来自 BERT 的 vocab_size（≈ 21128）
        #   复用 BERT 词表的原因：学生和教师使用相同的 token id，保证输入一致
        #
        # embedding_dim：词向量维度，设为 256
        #   BERT 的 hidden_size 是 768，这里用 256 大幅减少参数量
        #   嵌入层参数量：21128 × 256 ≈ 5.4M vs BERT 的 21128 × 768 ≈ 16.2M
        #
        # hidden_size：LSTM 隐藏层维度，设为 256
        #   双向后输出维度为 512
        #
        # num_layers：LSTM 堆叠层数，设为 3
        #   层数越多，模型容量越大，效果越好，但训练越慢
        #   3 层是效果和速度的较好平衡点
        # ==================================================================
        self.vocab_size = bert_config.vocab_size
        self.embedding_dim = 256
        self.hidden_size = 256
        self.num_layers = 3

        # ==================================================================
        # 3- 搭建网络结构
        # ==================================================================

        # ==================================================================
        # 3.1- 词嵌入层（Embedding Layer）
        #
        # nn.Embedding(num_embeddings, embedding_dim)：
        #   本质是一个形状为 (vocab_size, embedding_dim) 的权重矩阵
        #   输入 token id，输出对应的词向量
        #
        # 原理：查找表（Lookup Table）
        #   输入 input_ids = [101, 123, 456]  → 第101、123、456行的向量
        #
        # 形状变化：
        #   输入:  (batch_size, seq_len) 整数索引
        #   输出:  (batch_size, seq_len, embedding_dim) 浮点向量
        #
        # 训练过程：词向量随机初始化，随着训练不断更新
        #   最终学到的词向量中，语义相近的词向量距离也近
        # ==================================================================
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim
        )

        # ==================================================================
        # 3.2- 双向 LSTM 层（BiLSTM）
        #
        # nn.LSTM 参数详解：
        #   input_size=self.embedding_dim (256)：
        #     输入特征维度，必须等于词向量维度
        #
        #   hidden_size=self.hidden_size (256)：
        #     隐藏层维度，LSTM 内部状态的维度
        #     双向时，输出维度 = hidden_size × 2 = 512
        #
        #   num_layers=self.num_layers (3)：
        #     堆叠的 LSTM 层数
        #     多层 LSTM 可以提取更抽象的特征
        #     注意：与双向无关，双向是在每层内部做两个方向
        #
        #   batch_first=True：
        #     True: 输入形状 (batch, seq, feature)
        #     False: 输入形状 (seq, batch, feature) —— PyTorch 默认
        #     设为 True 更符合直觉，也和 DataLoader 输出格式一致
        #
        #   bidirectional=True：
        #     使用双向 LSTM
        #     输出维度变为 hidden_size × 2 = 512
        #     参数数量也翻倍
        #
        # LSTM 内部自动初始化：
        #   如果不传 h_0 和 c_0，PyTorch 自动初始化为全零
        #   对应源代码：torch/nn/modules/rnn.py 的 LSTM.forward 方法
        #
        # 输出形状：
        #   output: (batch, seq_len, hidden_size × 2) → (batch, seq_len, 512)
        #   h_n:    (num_layers × 2, batch, hidden_size) → (6, batch, 256)
        #   c_n:    (num_layers × 2, batch, hidden_size) → (6, batch, 256)
        # ==================================================================
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )

        # ==================================================================
        # 3.3- Dropout 层（随机失活）
        #
        # nn.Dropout(p=0.3)：
        #   训练时随机将 30% 的神经元输出置为 0
        #   推理时自动关闭（model.eval() 时失效）
        #
        # 作用：防止过拟合
        #   强迫模型不依赖某几个特定神经元，学习更鲁棒的特征
        #
        # p 值选择：
        #   p=0.3：中等强度的正则化，适合本模型规模
        #   p=0.5：较强正则化，大模型常用
        #   p=0.1：较弱正则化，小模型或数据少时用
        # ==================================================================
        self.dropout = nn.Dropout(0.3)

        # ==================================================================
        # 3.4- 全连接分类层（Linear）
        #
        # nn.Linear(in_features, out_features)：
        #   将池化后的句子向量映射到类别空间
        #
        # in_features=hidden_size × 2 = 512：
        #   因为是双向 LSTM，正向和反向的隐藏状态拼接后维度翻倍
        #   正向 LSTM 输出 256 维 + 反向 LSTM 输出 256 维 → 拼接为 512 维
        #
        # out_features=config.class_cnt：
        #   输出类别数量的 logits
        #
        # 形状变化：
        #   输入:  (batch_size, 512)
        #   输出:  (batch_size, class_cnt)
        # ==================================================================
        self.linear = nn.Linear(
            in_features=self.hidden_size * 2,
            out_features=config.class_cnt
        )

    def forward(self, input_ids, attention_mask):
        """
        前向传播：完整的推理流程。

        数据流：
            input_ids → Embedding → 掩码过滤 → BiLSTM → 平均池化 → Dropout → Linear → logits

        参数:
            input_ids (Tensor):       词索引张量，形状 (batch_size, max_length)
                                      [CLS]=101, [SEP]=102, [PAD]=0, 其他=正常词id
            attention_mask (Tensor):  注意力掩码，形状 (batch_size, max_length)
                                      1=有效token，0=padding填充

        返回:
            Tensor: 分类 logits，形状 (batch_size, class_cnt)
                    未经过 softmax，可直接输入 CrossEntropyLoss
        """
        # ==================================================================
        # 步骤1：词嵌入
        #
        # self.embedding(input_ids)：
        #   将整数 token id 转换为稠密向量
        #   形状变化：(batch, seq_len) → (batch, seq_len, embedding_dim)
        #   例：(32, 32) → (32, 32, 256)
        # ==================================================================
        ebd = self.embedding(input_ids)

        # ==================================================================
        # 步骤2：构建掩码，过滤特殊 token
        #
        # 目标：只让有实际语义的词参与后续计算，过滤掉三类特殊 token：
        #   - [CLS] (101)：BERT 添加的句子开头标记，分类时有用，但 LSTM 不需要
        #   - [SEP] (102)：BERT 添加的句子结尾标记，LSTM 不需要
        #   - [PAD] (0)：   填充标记，无实际意义
        #
        # 掩码构建逻辑（布尔运算）：
        #   第一步：input_ids != 101   → 标记非 [CLS] 位置
        #   第二步：input_ids != 102   → 标记非 [SEP] 位置
        #   第三步：& 运算             → 同时满足两个条件（既不是 [CLS] 也不是 [SEP]）
        #   第四步：& attention_mask   → 再过滤掉 padding 位置
        #
        # 具体例子（input_ids 形状 2×5）：
        #   input_ids     = [[101, 123, 456, 102,   0],     # [CLS] 词1 词2 [SEP] [PAD]
        #                     [101, 789, 102,   0,   0]]
        #   第一步后       = [[F, T, T, T, T], [F, T, T, T, T]]
        #   第二步后       = [[T, T, T, F, T], [T, T, F, T, T]]
        #   & 后           = [[F, T, T, F, T], [F, T, F, T, T]]
        #   & attention后  = [[F, T, T, F, F], [F, T, F, F, F]]  ← 只有词1词2保留
        # ==================================================================
        cls_token_index = 101
        sep_token_index = 102

        # 过滤 [CLS] 和 [SEP]：两个条件同时满足才为 True
        ebd_mask = (input_ids != cls_token_index) & (input_ids != sep_token_index)

        # 过滤 [PAD]：attention_mask 中为 0 的位置就是 padding
        # attention_mask.bool() 将 1→True, 0→False，然后 & 运算
        ebd_mask = ebd_mask & attention_mask.bool()

        # ==================================================================
        # 步骤3：掩码升维 & 应用掩码
        #
        # ebd_mask.unsqueeze(-1)：
        #   在最后一个维度增加一维
        #   形状变化：(batch, seq_len) → (batch, seq_len, 1)
        #   目的：与 ebd 的 (batch, seq_len, embedding_dim) 进行广播乘法
        #
        # ebd * ebd_mask：
        #   PyTorch 广播机制：将 (batch, seq_len, 1) 自动扩展为 (batch, seq_len, embedding_dim)
        #   掩码值为 True(1) 的位置：词向量保持不变
        #   掩码值为 False(0) 的位置：词向量变为全零向量
        #
        # 为什么用乘法而不是直接切片删除？
        #   - 保持张量形状不变，后续 LSTM 需要固定长度的输入
        #   - 全零向量在 LSTM 中不影响有效词的计算
        # ==================================================================
        ebd_mask = ebd_mask.unsqueeze(-1)
        ebd = ebd * ebd_mask

        # ==================================================================
        # 步骤4：BiLSTM 前向传播
        #
        # self.lstm(ebd) 返回 (output, (h_n, c_n))：
        #   output: (batch, seq_len, hidden_size × 2)  所有时间步的隐藏状态
        #   h_n:    最后一层所有方向的隐藏状态（这里不需要）
        #   c_n:    最后一层所有方向的细胞状态（这里不需要）
        #
        # 为什么不需要手动传 h_0 和 c_0？
        #   PyTorch 的 LSTM 内部自动初始化为全零
        #   对应源代码：torch/nn/modules/rnn.py 第 1056 行附近
        #   全零初始化在大多数情况下效果足够好
        # ==================================================================
        output, (h_n, c_n) = self.lstm(ebd)

        # ==================================================================
        # 步骤5：平均池化（Average Pooling）
        #
        # 目标：将变长的句子级信息压缩为固定维度的向量，用于分类
        #
        # 为什么用 sum(dim=1)？
        #   - output 形状：(batch, seq_len, hidden_size × 2)
        #   - 分类任务需要的是"整句话"的语义，而非"每个词"的语义
        #   - sum(dim=1) 沿 seq_len 维度求和，得到 (batch, hidden_size × 2)
        #   - 结果含义：句子中所有词向量的总和
        #
        # 分子 output_sum：所有有效词的向量之和
        #   之前全零向量（被掩码的位置）求和贡献为 0，不影响结果
        #
        # 分母 token_cnt：每个句子中有效词的数量
        #   ebd_mask.sum(dim=1)：统计每个句子有多少个 True（即有效词）
        #   + 1e-6：防止除以 0（极端情况：句子全是特殊 token）
        #
        # 为什么用平均值而不是直接求和？
        #   - 消除句子长度的影响：长句子求和值大，短句子求和值小，不公平
        #   - 平均值统一了尺度，让模型更关注词的语义而非句子长度
        #   - 避免某几个异常词主导整个句子表示
        #
        # 其他池化方式对比：
        #   - 最大池化：取每个维度最大值，保留最显著特征，但对噪声敏感
        #   - 平均池化：取所有词的平均，平滑稳定，适合短文本分类
        #   - [CLS] 向量：取第一个 token，需要模型学会在 [CLS] 中汇总信息
        # ==================================================================
        output_sum = output.sum(dim=1)                      # (batch, hidden_size*2)
        token_cnt = ebd_mask.sum(dim=1) + 1e-6              # (batch, 1)
        output = output_sum / token_cnt                      # (batch, hidden_size*2)

        # ==================================================================
        # 步骤6：Dropout + 全连接分类
        #
        # self.dropout(output)：训练时随机丢弃 30% 神经元，防止过拟合
        # self.linear(...)：将 512 维句子向量映射到 class_cnt 维 logits
        #
        # 形状变化：
        #   Dropout 输入:  (batch, 512)
        #   Dropout 输出:  (batch, 512)（形状不变，部分值变 0）
        #   Linear 输出:   (batch, class_cnt)
        #
        # 返回 logits（未归一化），由外部损失函数处理：
        #   - 硬标签损失：CrossEntropyLoss(logits, labels)
        #   - 软标签损失：KLDivLoss(softmax(logits/T), teacher_soft_labels)
        # ==================================================================
        return self.linear(self.dropout(output))


# ==========================================================================
# 测试入口：验证模型结构、参数量、前向传播
# ==========================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("【知识蒸馏 - 学生模型（BiLSTM）测试】")
    print("=" * 70)

    # 1- 实例化学生模型
    model = BiLSTMStuModel()
    print(f"\n模型结构:\n{model}")

    # 2- 统计参数量（与学生模型对比）
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"类别数量: {config.class_cnt}")
    print(f"使用设备: {config.device}")

    # 3- 测试前向传播（模拟数据）
    print("\n" + "=" * 70)
    print("【前向传播测试】")
    print("=" * 70)

    batch_size = 2
    # 模拟 input_ids：包含 [CLS](101)、正常词、[SEP](102)、[PAD](0)
    dummy_input_ids = torch.tensor([
        [101, 1234, 5678, 102, 0, 0],     # 句子1：[CLS] 词1 词2 [SEP] [PAD] [PAD]
        [101, 9999, 102, 0, 0, 0]          # 句子2：[CLS] 词1 [SEP] [PAD] [PAD] [PAD]
    ])
    # 模拟 attention_mask：1=有效，0=padding
    dummy_attention_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0]
    ])

    # 前向传播
    with torch.no_grad():
        logits = model(dummy_input_ids, dummy_attention_mask)
        print(f"输入 input_ids 形状: {dummy_input_ids.shape}")
        print(f"输入 attention_mask 形状: {dummy_attention_mask.shape}")
        print(f"输出 logits 形状: {logits.shape}")
        print(f"输出 logits 值:\n{logits}")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)