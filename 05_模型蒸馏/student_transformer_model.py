"""
====================================================================================================
student_transformer_model.py —— 知识蒸馏：学生模型（小 Transformer）定义
====================================================================================================
【文件作用】
    定义知识蒸馏中的"学生模型"（Student Model）—— 3 层 Transformer 分类模型。
    与教师模型（BERT 12层）同构，支持软标签蒸馏 + 中间层蒸馏，效果优于 BiLSTM 学生。

【同构蒸馏核心知识点】
----------------------------------------------------------------------------------------------------
1. 为什么能用 Transformer 做学生模型？
   - 同构优势：师生结构相同，中间层语义可直接对齐，支持中间层蒸馏
   - 参数量可控：3 层 Transformer 约 15M 参数，是 BERT 12 层（110M）的 1/7
   - 效果更好：同构蒸馏通常比异构蒸馏（如 BiLSTM）效果更好
   - 工业界实践：DistilBERT、TinyBERT 都是同构蒸馏的代表

2. 三层蒸馏体系（本模型支持）
   ┌──────────────────────┬──────────────────────────────────────────────┐
   │ 软标签蒸馏             │ 学生输出概率 ≈ 教师输出概率（带温度 T）         │
   │ Response-based        │ 损失：KL(student_logits/T, teacher_logits/T) │
   ├──────────────────────┼──────────────────────────────────────────────┤
   │ 中间层蒸馏             │ 学生每层隐藏状态 ≈ 教师对应层隐藏状态           │
   │ Feature-based         │ 损失：MSE(student_hidden, teacher_hidden)     │
   ├──────────────────────┼──────────────────────────────────────────────┤
   │ 硬标签损失             │ 学生输出 ≈ 真实标签                           │
   │ Hard Label            │ 损失：CrossEntropyLoss(student_logits, label) │
   └──────────────────────┴──────────────────────────────────────────────┘

   总损失 = α × 硬标签损失 + β × 软标签损失 + γ × 中间层损失

3. 中间层蒸馏的层映射策略
   教师 12 层 → 学生 3 层，每 4 层教师对应 1 层学生：

   ┌──────────────────────┬──────────────────────────────────────────────┐
   │ 学生第 1 层           │ 学习教师第 4 层（底层语义：词法、浅层语法）        │
   │ 学生第 2 层           │ 学习教师第 8 层（中层语义：短语、依存关系）        │
   │ 学生第 3 层           │ 学习教师第 12 层（高层语义：句子级抽象）          │
   └──────────────────────┴──────────────────────────────────────────────┘

   为什么是 4/8/12 而不是 1/2/3？
   - 底层 Transformer 层处理局部特征（词法），中层处理短语级特征，高层处理全局语义
   - 跳过底层让学生先学"更重要的"高层语义，效果反而更好
   - 这是 TinyBERT 论文验证过的策略

4. 维度对齐（Projection Layer）
   教师 hidden_size = 768，学生 d_model = 312
   学生每层输出的 312 维需要投影到 768 维才能和教师计算 MSE：

       学生输出 (batch, seq_len, 312) → Linear(312, 768) → (batch, seq_len, 768)
       教师输出 (batch, seq_len, 768) → 直接使用

   每层一个投影层，共 3 个投影层，参数量：312 × 768 × 3 ≈ 0.7M

5. 同构蒸馏 vs 异构蒸馏
   ┌──────────────────┬──────────────────────────────────────────────────┐
   │ 同构蒸馏（本模型）  │ 师生结构相同，中间层可对齐，效果更好                 │
   │                  │ 代表：DistilBERT、TinyBERT                        │
   │                  │ 参数量 ~15M，F1 可达教师的 95%+                    │
   ├──────────────────┼──────────────────────────────────────────────────┤
   │ 异构蒸馏（BiLSTM） │ 师生结构不同，只能做软标签蒸馏                       │
   │                  │ 参数量 ~7M，F1 约教师的 85-90%                     │
   │                  │ 优势：推理更快，部署更简单                           │
   └──────────────────┴──────────────────────────────────────────────────┘

6. Transformer Encoder 层核心结构
   每层包含两个子层（与 BERT 完全相同）：
   ┌──────────────────────────────────────────────────────────────────┐
   │ 子层1：多头自注意力（Multi-Head Self-Attention）                    │
   │   Q = X @ W_Q, K = X @ W_K, V = X @ W_V                         │
   │   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V            │
   │   作用：让每个词关注句子中所有其他词，捕获全局依赖                      │
   ├──────────────────────────────────────────────────────────────────┤
   │ 子层2：前馈网络（Feed-Forward Network）                            │
   │   FFN(x) = ReLU(x @ W_1 + b_1) @ W_2 + b_2                      │
   │   作用：对每个位置独立做非线性变换，增强表达能力                        │
   ├──────────────────────────────────────────────────────────────────┤
   │ 每个子层后：残差连接 + LayerNorm                                    │
   │   output = LayerNorm(x + Sublayer(x))                            │
   │   作用：缓解梯度消失，加速训练                                        │
   └──────────────────────────────────────────────────────────────────┘

7. 位置编码（Positional Encoding）
   - Transformer 本身不感知顺序，需要额外注入位置信息
   - 本模型使用可学习的位置编码（learned positional embedding）
   - 最大长度 = max_length（来自 config，默认 32）
   - 与 BERT 的 position embedding 原理一致

【依赖包说明】
----------------------------------------------------------------------------------------------------
1. torch
   - torch.nn:                         神经网络模块
   - torch.nn.functional:              函数式 API（softmax, relu 等）
   - torch.nn.TransformerEncoderLayer: PyTorch 内置的 Transformer 编码器层
   - torch.nn.TransformerEncoder:      堆叠多层 TransformerEncoderLayer

2. nn.TransformerEncoderLayer 参数详解
   - d_model:          输入输出维度（= 312）
   - nhead:            多头注意力头数（= 12，需整除 d_model）
   - dim_feedforward:  前馈网络隐藏层维度（= 1200，通常 4×d_model）
   - dropout:          Dropout 概率
   - activation:       激活函数（默认 'relu'）
   - batch_first:      True 时输入形状 (batch, seq, feature)

3. nn.TransformerEncoder 参数
   - encoder_layer:    TransformerEncoderLayer 实例
   - num_layers:       堆叠层数（= 3）

【面试高频考点 ⭐】
    1. 同构蒸馏和异构蒸馏的区别？各自的优缺点？
    2. 中间层蒸馏的原理？层映射策略怎么选？
    3. Transformer 的自注意力机制原理？Q、K、V 分别是什么？
    4. 为什么需要位置编码？可学习 vs 正弦位置编码的区别？
    5. 残差连接和 LayerNorm 的作用？为什么放在子层后面？
    6. 多头注意力的"多头"是什么意思？为什么比单头好？
    7. 投影层（Projection）的作用？为什么需要维度对齐？
    8. DistilBERT 和 TinyBERT 的蒸馏策略有什么不同？
    9. 如何选择 distill_layers 的映射关系？
    10. 三层蒸馏的总损失函数怎么设计？
    11. 为什么中间层蒸馏用 MSE 而不是 KL 散度？
    12. Transformer 学生和 BiLSTM 学生各适合什么场景？
====================================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from transformers import BertConfig

# ==========================================================================
# 全局变量
# ==========================================================================
config = Config()

# 复用 BERT 词表
bert_config = BertConfig.from_pretrained(config.bert_path)


# ==========================================================================
# 类：TransformerStuModel
# 作用：知识蒸馏的学生模型（3 层 Transformer 分类模型）
#
# ==========================================================================
# 【模型蒸馏知识点：同构蒸馏的设计原则】
#   1. 结构同构：师生都使用 Transformer Encoder，中间层语义可直接对齐
#   2. 层数压缩：12 层 → 3 层，每 4 层教师对应 1 层学生
#   3. 维度压缩：768 → 312，减少参数量但保留足够的表达能力
#   4. 三层蒸馏：硬标签 + 软标签 + 中间层，三管齐下
# ==========================================================================
class TransformerStuModel(nn.Module):
    """
    知识蒸馏学生模型：3 层 Transformer Encoder + 平均池化 + 全连接分类头。

    架构：
        input_ids → Embedding + Position Encoding → Transformer × 3
        → 平均池化 → Dropout → Linear → logits

    额外输出（用于中间层蒸馏）：
        每层 Transformer 的隐藏状态 → 投影层 → 与教师对应层对齐

    参数量：约 15M（BERT 110M 的 1/7）

    继承关系：
        nn.Module，必须实现 __init__() 和 forward()
    """

    def __init__(self, return_hidden=False):
        """
        初始化学生 Transformer 模型。

        参数:
            return_hidden (bool):  是否返回中间层隐藏状态
                                   True：训练时用，返回所有层输出用于中间层蒸馏
                                   False：推理时用，只返回 logits
        """
        # ==================================================================
        # 1- 初始化父类
        # ==================================================================
        super().__init__()

        # ==================================================================
        # 2- 设置超参数
        #
        # d_model = 312：
        #   模型隐藏维度，选择 312 的原因：
        #   - 312 / 12 = 26，能被注意力头数整除
        #   - 约 BERT 768 的 40%，参数量约为 BERT 的 16%
        #   - 太小（如 128）表达能力不足，太大（如 512）参数量接近 BERT 失去蒸馏意义
        #
        # nhead = 12：
        #   多头注意力头数，和 BERT 保持一致
        #   每头维度 = 312 / 12 = 26
        #
        # dim_feedforward = 1200：
        #   前馈网络隐藏层维度，约 4 × d_model = 1248，取整为 1200
        #
        # num_layers = 3：
        #   Transformer 层数，是 BERT 12 层的 1/4
        #   三层分别对应教师模型的底层、中层、高层
        #
        # max_position_embeddings = 32：
        #   最大位置编码长度，与 config.max_length 一致
        # ==================================================================
        self.vocab_size = bert_config.vocab_size        # 21128
        self.d_model = 312                               # 隐藏维度
        self.nhead = 12                                  # 注意力头数
        self.dim_feedforward = 1200                      # FFN 隐藏层维度
        self.num_layers = 3                              # Transformer 层数
        self.max_position_embeddings = config.max_length  # 最大位置编码长度
        self.return_hidden = return_hidden               # 是否返回中间层

        # ==================================================================
        # 3- 搭建网络结构
        # ==================================================================

        # ==================================================================
        # 3.1- 词嵌入层（Token Embedding）
        #
        # nn.Embedding(vocab_size, d_model)：
        #   将 token id 映射为 d_model 维的稠密向量
        #   参数量：21128 × 312 ≈ 6.6M
        #   这是模型参数的主要部分
        # ==================================================================
        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model
        )

        # ==================================================================
        # 3.2- 位置编码（Position Embedding）
        #
        # 为什么需要位置编码？
        #   Transformer 的自注意力机制是"无序"的——它同时看所有位置，
        #   无法区分"我打你"和"你打我"。位置编码注入顺序信息。
        #
        # 可学习位置编码：
        #   每个位置随机初始化一个向量，训练中学习
        #   优点：灵活，能适应不同任务的位置模式
        #   缺点：不能外推到训练时没见过的长度
        #
        # BERT 也使用可学习位置编码，与这里一致
        #
        # 形状：(max_position_embeddings, d_model) = (32, 312)
        # ==================================================================
        self.position_embedding = nn.Embedding(
            num_embeddings=self.max_position_embeddings,
            embedding_dim=self.d_model
        )

        # ==================================================================
        # 3.3- LayerNorm + Dropout（嵌入后）
        #
        # nn.LayerNorm(d_model)：
        #   对每个样本的 d_model 维度做归一化
        #   公式：y = (x - mean) / sqrt(var + eps) * γ + β
        #   作用：稳定训练，加速收敛
        #
        # BERT 在 Embedding 后也加了 LayerNorm + Dropout
        # ==================================================================
        self.embedding_norm = nn.LayerNorm(self.d_model)
        self.embedding_dropout = nn.Dropout(0.1)

        # ==================================================================
        # 3.4- Transformer 编码器层
        #
        # nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first)：
        #   单层 Transformer Encoder，包含：
        #     1. Multi-Head Self-Attention
        #     2. Add & LayerNorm（残差连接 + 层归一化）
        #     3. Feed-Forward Network（两个线性层 + ReLU）
        #     4. Add & LayerNorm
        #
        # nn.TransformerEncoder(encoder_layer, num_layers)：
        #   堆叠多层 TransformerEncoderLayer
        #   每层输出会作为下一层的输入
        # ==================================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=0.1,
            activation='relu',
            batch_first=True         # 输入形状 (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers
        )

        # ==================================================================
        # 3.5- 中间层投影层（用于中间层蒸馏）
        #
        # 为什么需要投影层？
        #   学生 d_model=312，教师 hidden_size=768，维度不匹配
        #   需要将学生每层输出投影到 768 维，才能和教师计算 MSE 损失
        #
        # 每层一个独立的投影层（共 3 个）：
        #   nn.Linear(312, 768)
        #   参数量：312 × 768 × 3 ≈ 0.7M
        #
        # 投影层只在训练时使用，推理时不需要
        # ==================================================================
        self.distill_projections = nn.ModuleList([
            nn.Linear(self.d_model, bert_config.hidden_size)  # 312 → 768
            for _ in range(self.num_layers)
        ])

        # ==================================================================
        # 3.6- 最终池化层 + 分类头
        #
        # 与 BiLSTM 学生一样，使用平均池化
        # 但因为 Transformer 有 [PAD] 掩码，需要在池化时忽略 padding
        #
        # 分类头：d_model → class_cnt
        # ==================================================================
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.d_model, config.class_cnt)

    def forward(self, input_ids, attention_mask):
        """
        前向传播。

        数据流：
            input_ids → Token Embedding + Position Embedding
            → LayerNorm + Dropout
            → Transformer Encoder × 3
            → 平均池化（忽略 padding）
            → Dropout → Linear → logits

        参数:
            input_ids (Tensor):       词索引，形状 (batch_size, max_length)
            attention_mask (Tensor):  掩码，形状 (batch_size, max_length)
                                      1=有效token，0=padding，用于忽略 padding

        返回:
            如果 return_hidden=True（训练时）：
                (logits, hidden_states_list)
                - logits: (batch, class_cnt) 分类 logits
                - hidden_states_list: list[3 个 Tensor]，每层投影后的隐藏状态
                                      每个形状 (batch, seq_len, 768)
            如果 return_hidden=False（推理时）：
                logits: (batch, class_cnt)
        """
        batch_size, seq_len = input_ids.shape

        # ==================================================================
        # 步骤1：Token Embedding + Position Embedding
        #
        # token_embedding: 将 token id 转为词向量
        # position_embedding: 为每个位置生成位置编码
        # 两者相加得到最终嵌入（与 BERT 一致）
        # ==================================================================
        # 生成位置索引：[0, 1, 2, ..., seq_len-1]，扩展到 batch 维度
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        token_emb = self.token_embedding(input_ids)           # (batch, seq_len, d_model)
        pos_emb = self.position_embedding(position_ids)       # (batch, seq_len, d_model)
        hidden_states = token_emb + pos_emb                    # (batch, seq_len, d_model)

        # LayerNorm + Dropout
        hidden_states = self.embedding_norm(hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        # ==================================================================
        # 步骤2：构建 padding 掩码（给 Transformer 用）
        #
        # Transformer 需要 src_key_padding_mask 来忽略 padding 位置
        # 形状：(batch, seq_len)，True 表示该位置需要被忽略
        #
        # attention_mask 中 1=有效，0=padding
        # 取反后：False=有效，True=padding（需要忽略）
        # ==================================================================
        src_key_padding_mask = (attention_mask == 0)  # (batch, seq_len)

        # ==================================================================
        # 步骤3：Transformer Encoder 前向传播
        #
        # 如果 return_hidden=True，需要逐层执行以捕获中间层输出
        # 如果 return_hidden=False，直接调用整个 encoder（更快）
        # ==================================================================
        if self.return_hidden:
            # 逐层执行，收集每层输出用于中间层蒸馏
            all_hidden_states = []

            for layer_idx, layer in enumerate(self.transformer_encoder.layers):
                # 单层 Transformer 前向传播
                hidden_states = layer(
                    hidden_states,
                    src_key_padding_mask=src_key_padding_mask
                )
                # 投影到教师维度，用于中间层蒸馏
                projected = self.distill_projections[layer_idx](hidden_states)
                all_hidden_states.append(projected)

            # 最终输出是最后一层的结果
            encoder_output = hidden_states  # (batch, seq_len, d_model)
        else:
            # 推理模式：直接调用整个 encoder，不需要中间层输出
            encoder_output = self.transformer_encoder(
                hidden_states,
                src_key_padding_mask=src_key_padding_mask
            )  # (batch, seq_len, d_model)

        # ==================================================================
        # 步骤4：平均池化（忽略 padding 位置）
        #
        # 原理：对 seq_len 维度求平均，但只计算有效 token
        # 分子：所有位置的向量求和（padding 位置已经在 attention 中被忽略）
        # 分母：有效 token 数量
        #
        # 注意：Transformer 的自注意力机制中，padding 位置的输出
        # 虽然也会被计算，但因为 attention 权重为 0（被 mask），
        # 所以 padding 位置的输出实际上接近 0 向量
        # 为了更精确，我们还是用 attention_mask 来加权平均
        # ==================================================================
        # attention_mask 升维：(batch, seq_len) → (batch, seq_len, 1)
        mask_expanded = attention_mask.unsqueeze(-1).float()

        # 加权求和：padding 位置 mask=0，不贡献
        encoder_output = encoder_output * mask_expanded  # padding 位置变 0

        # 求和 + 求平均
        output_sum = encoder_output.sum(dim=1)                      # (batch, d_model)
        token_cnt = mask_expanded.sum(dim=1) + 1e-6                 # (batch, 1)
        pooled_output = output_sum / token_cnt                       # (batch, d_model)

        # ==================================================================
        # 步骤5：Dropout + 全连接分类
        # ==================================================================
        logits = self.linear(self.dropout(pooled_output))  # (batch, class_cnt)

        # ==================================================================
        # 返回：训练时同时返回 logits 和中间层隐藏状态
        # ==================================================================
        if self.return_hidden:
            return logits, all_hidden_states
        else:
            return logits


# ==========================================================================
# 测试入口：验证模型结构、参数量、前向传播、中间层输出
# ==========================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("【知识蒸馏 - 学生模型（3层 Transformer）测试】")
    print("=" * 70)

    # 1- 实例化学生模型（训练模式：返回中间层）
    model = TransformerStuModel(return_hidden=True)
    print(f"\n模型结构:\n{model}")

    # 2- 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"  对比：BERT 教师 ≈ 110M，BiLSTM 学生 ≈ 7M，Transformer 学生 ≈ 15M")
    print(f"  类别数量: {config.class_cnt}")
    print(f"  d_model: {model.d_model}, nhead: {model.nhead}, layers: {model.num_layers}")

    # 3- 测试前向传播（模拟数据）
    print("\n" + "=" * 70)
    print("【前向传播测试】")
    print("=" * 70)

    batch_size = 2
    dummy_input_ids = torch.tensor([
        [101, 1234, 5678, 102, 0, 0],
        [101, 9999, 102, 0, 0, 0]
    ])
    dummy_attention_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0]
    ])

    with torch.no_grad():
        logits, hidden_states = model(dummy_input_ids, dummy_attention_mask)
        print(f"输入 input_ids 形状: {dummy_input_ids.shape}")
        print(f"输入 attention_mask 形状: {dummy_attention_mask.shape}")
        print(f"输出 logits 形状: {logits.shape}")

        print(f"\n中间层隐藏状态（用于中间层蒸馏）：")
        for i, hs in enumerate(hidden_states):
            print(f"  第{i+1}层投影后形状: {hs.shape}  (batch, seq_len, 768)")

    # 4- 测试推理模式（不返回中间层）
    print("\n" + "=" * 70)
    print("【推理模式测试】")
    print("=" * 70)

    model_infer = TransformerStuModel(return_hidden=False)
    with torch.no_grad():
        logits_only = model_infer(dummy_input_ids, dummy_attention_mask)
        print(f"推理模式输出 logits 形状: {logits_only.shape}")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)