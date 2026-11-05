"""
====================================================================================================
teacher_bert_model.py —— 知识蒸馏：教师模型（BERT）定义
====================================================================================================
【文件作用】
    定义知识蒸馏中的"教师模型"（Teacher Model）。
    教师模型是一个完全训练好的 BERT 分类模型，用于生成"软标签"（soft labels），
    指导"学生模型"（Student Model，一个更小的模型）学习。

【知识蒸馏核心概念】
----------------------------------------------------------------------------------------------------
1. 什么是知识蒸馏（Knowledge Distillation）？
   - 一种模型压缩技术，将大模型（教师）学到的知识迁移到小模型（学生）
   - 由 Hinton 等人在 2015 年提出（论文: "Distilling the Knowledge in a Neural Network"）
   - 核心思想：教师模型输出的概率分布（软标签）包含比 one-hot 标签（硬标签）更丰富的信息

2. 为什么要做知识蒸馏？
   - 部署需求：BERT-base 约 110M 参数，推理速度慢，不适合移动端/实时场景
   - 效果保留：小模型直接训练效果差，但通过蒸馏可以接近大模型的效果
   - 成本优化：推理时用小模型，降低计算成本和延迟

3. 硬标签 vs 软标签（核心区别）
   ┌──────────────────┬──────────────────────────────────────────────────────┐
   │ 硬标签（Hard Label） │ one-hot 形式，如 [0, 0, 1, 0, 0]                    │
   │                    │ 只告诉模型"正确答案是第3类"                            │
   │                    │ 丢失了类别之间的相似性信息                             │
   ├──────────────────┼──────────────────────────────────────────────────────┤
   │ 软标签（Soft Label） │ 概率分布，如 [0.02, 0.05, 0.85, 0.06, 0.02]         │
   │                    │ 告诉模型"第3类最可能，第4类次之，第1类最不可能"          │
   │                    │ 保留了类别之间的相似性（暗知识）                        │
   └──────────────────┴──────────────────────────────────────────────────────┘

4. 温度系数 T（Temperature）的作用
   - 公式：soft_label = softmax(logits / T)
   - T=1：标准 softmax，概率分布尖锐（高概率类别接近1，其他接近0）
   - T>1：软化概率分布，让类别间差异变小，暗知识更明显
   - 典型值：T=4~20，T 越大分布越平滑
   - 原理：高温让教师模型"不那么自信"，暴露更多类别间的相似性信息

5. 蒸馏损失函数
   总损失 = α × 硬标签损失 + (1-α) × 软标签损失
   - 硬标签损失（Hard Loss）：学生输出 vs 真实标签（CrossEntropyLoss）
   - 软标签损失（Soft Loss）：学生输出/T vs 教师输出/T（KL散度 或 MSE）
   - α：平衡系数，通常 0.1~0.5
   - 软标签损失需要乘以 T² 来缩放（保持梯度大小一致）

6. 蒸馏流程
   第1步：训练教师模型（大模型，如 BERT）→ 得到高精度
   第2步：用教师模型对所有训练数据生成软标签（概率分布）
   第3步：训练学生模型（小模型，如 TextCNN/LSTM），同时用硬标签和软标签监督
   第4步：部署学生模型（推理速度快，效果接近教师）

7. 常见的教师-学生组合
   ┌──────────────────────┬──────────────────────────────────────────────┐
   │ 教师模型              │ 学生模型                                      │
   ├──────────────────────┼──────────────────────────────────────────────┤
   │ BERT-base (110M)     │ TextCNN / BiLSTM / 3层Transformer (~1-5M)    │
   │ BERT-large (340M)    │ BERT-tiny / DistilBERT (~15-66M)             │
   │ GPT-4                │ GPT-3.5 / LLaMA-7B                            │
   └──────────────────────┴──────────────────────────────────────────────┘

8. 蒸馏 vs 其他模型压缩技术
   ┌──────────────────┬───────────────────────────────────────────────────┐
   │ 知识蒸馏          │ 用大模型教小模型，需要训练学生模型                     │
   ├──────────────────┼───────────────────────────────────────────────────┤
   │ 模型量化          │ 将 FP32 权重转为 INT8/INT4，减少模型体积和推理时间     │
   │                  │ 不需要重新训练，但可能有精度损失                       │
   ├──────────────────┼───────────────────────────────────────────────────┤
   │ 模型剪枝          │ 移除不重要的权重/神经元，减少参数量                     │
   │                  │ 需要微调恢复精度                                     │
   ├──────────────────┼───────────────────────────────────────────────────┤
   │ 低秩分解          │ 将大矩阵分解为小矩阵乘积（如 SVD），减少参数量           │
   └──────────────────┴───────────────────────────────────────────────────┘

【依赖包说明】
----------------------------------------------------------------------------------------------------
1. torch
   PyTorch 深度学习核心库。
   常用 API：
   - torch.tensor(data, dtype=...)     从列表/数组创建张量
   - torch.cuda.is_available()         检查 GPU 是否可用
   - torch.save(obj, path)             保存模型/张量
   - torch.load(path)                  加载模型/张量
   - torch.no_grad()                   上下文管理器，禁用梯度计算
   - torch.softmax(input, dim)         沿指定维度计算 softmax
   - torch.argmax(input, dim)          返回指定维度上最大值的索引

2. torch.nn（神经网络模块）
   - nn.Module:                        所有神经网络模型的基类
   - nn.Linear(in_features, out_features):  全连接层
   - nn.functional.softmax(input, dim):     softmax 函数式版本
   - nn.functional.kl_div():                KL 散度（蒸馏损失常用）

3. transformers（HuggingFace 预训练模型库）
   - BertModel：BERT 预训练模型
       输入：input_ids, attention_mask
       输出：BaseModelOutputWithPooling
         - last_hidden_state:  (batch, seq_len, 768)
         - pooler_output:      (batch, 768) [CLS] 经过 Linear+Tanh

   - BertConfig：BERT 模型配置
       常用属性：
         - hidden_size:        768（BERT-base）
         - num_hidden_layers:  12
         - num_attention_heads: 12

【面试高频考点 ⭐】
    1. 什么是知识蒸馏？为什么需要它？
    2. 硬标签和软标签的区别？软标签为什么更好？
    3. 温度系数 T 的作用？T 越大分布越怎样？
    4. 蒸馏损失函数由哪两部分组成？各自的权重？
    5. 为什么软标签损失要乘以 T²？
    6. 教师模型和学生模型的关系？教师模型需要训练吗？
    7. KL 散度的含义？在蒸馏中如何使用？
    8. 知识蒸馏和模型量化、剪枝的区别？
    9. pooler_output 和 last_hidden_state[:,0] 的区别？蒸馏时用哪个？
    10. 蒸馏时教师模型参数是否需要冻结？为什么？
    11. 蒸馏后的学生模型真的能接近教师模型效果吗？
    12. 温度 T 的典型取值范围？如何选择合适的 T？
====================================================================================================
"""
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

# ==========================================================================
# 全局变量
# ==========================================================================
config = Config()


# ==========================================================================
# 类：TeacherBertModel
# 作用：知识蒸馏的教师模型（BERT 分类模型）
#
# ==========================================================================
# 【模型蒸馏知识点：教师模型的角色】
#   教师模型需要满足两个条件：
#   1. 训练充分，效果优秀（教师能力强，学生才能学得好）
#   2. 推理时参数冻结（教师不参与训练，只负责生成软标签）
#
#   教师模型的使用方式：
#   - 训练阶段：用教师模型对训练数据生成软标签，然后训练学生模型
#   - 部署阶段：只部署学生模型，教师模型不部署
#   - 教师模型可以离线预先对全部数据生成软标签，节省训练时间
# ==========================================================================
class TeacherBertModel(nn.Module):
    """
    知识蒸馏教师模型：BERT + 全连接分类头。

    架构：
        BERT（冻结参数）→ pooler_output → 全连接层 → logits
        → softmax(T) → 软标签（概率分布）

    与普通 BERT 分类模型的区别：
        1. 使用 pooler_output 而非 last_hidden_state[:,0]
           （pooler_output 包含更多语义信息，适合生成软标签）
        2. 提供 get_soft_labels() 方法，带温度系数的 softmax
        3. 参数冻结，只做推理，不参与训练

    继承关系：
        nn.Module 是所有神经网络模型的基类，必须实现：
        - __init__():  定义网络层
        - forward():   定义前向传播
    """

    def __init__(self):
        """
        初始化教师模型：加载 BERT + 定义分类头。

        关键步骤：
            1. super().__init__()：初始化 nn.Module 的参数追踪机制
            2. 加载预训练 BERT 模型
            3. 冻结 BERT 所有参数（教师模型不训练）
            4. 定义分类头（全连接层）
        """
        # ==================================================================
        # 1- 初始化父类
        #
        # super().__init__()：
        #   调用 nn.Module 的构造函数，初始化 PyTorch 内部的参数管理机制
        #   - _parameters: 存储模型参数
        #   - _modules: 存储子模块（如 self.bert_model, self.linear）
        #   - _buffers: 存储非参数的持久化张量
        #   如果不调用，后续 model.parameters() 等方法将无法正常工作
        # ==================================================================
        super().__init__()

        # ==================================================================
        # 2- 搭建网络结构
        # ==================================================================

        # ==================================================================
        # 2.1- 加载预训练 BERT 模型
        #
        # BertModel.from_pretrained(config.bert_path)：
        #   从本地路径加载预训练 BERT 模型
        #   BERT-base-chinese 结构：
        #     - Embedding 层：token embedding + position embedding + segment embedding
        #     - 12 层 Transformer Encoder（每层含多头自注意力 + 前馈网络）
        #     - Pooler 层：[CLS] 向量 → nn.Linear(768, 768) → Tanh
        #   总参数量：约 110M
        #
        # 教师模型选择 BERT 的原因：
        #   - 预训练充分，泛化能力强
        #   - 输出的概率分布质量高，包含丰富的"暗知识"
        #   - 参数量大，效果好，但推理慢（所以需要蒸馏）
        # ==================================================================
        self.bert_model = BertModel.from_pretrained(config.bert_path)

        # ==================================================================
        # 2.2- 冻结 BERT 参数（特征提取模式）
        #
        # 【模型蒸馏知识点：为什么要冻结教师模型？】
        #   1. 教师模型已经训练好了，蒸馏过程中不需要再更新
        #   2. 冻结参数可以：
        #      - 节省显存：不需要存储 BERT 的梯度（约 110M 参数 × 4 字节 ≈ 440MB）
        #      - 加速推理：不需要构建计算图，前向传播更快
        #      - 保证软标签稳定：教师参数不变，软标签一致，学生训练更稳定
        #   3. 如果教师模型也训练，就成了"联合训练"而非蒸馏，效果可能更差
        #
        # param.requires_grad_(False)：
        #   - requires_grad 属性控制是否需要计算梯度
        #   - False：反向传播跳过该参数，不更新
        #   - 末尾的 _ 表示原地操作（in-place），直接修改参数属性
        #
        # self.bert_model.parameters()：
        #   - 返回模型所有可训练参数的迭代器
        #   - 每个元素是 torch.nn.Parameter（Tensor 的子类）
        # ==================================================================
        for param in self.bert_model.parameters():
            param.requires_grad_(False)

        # ==================================================================
        # 2.3- 定义分类头（全连接层）
        #
        # BertConfig.from_pretrained(path).hidden_size：
        #   BERT-base 隐藏层维度 = 768
        #
        # nn.Linear(in_features, out_features)：
        #   全连接层，执行线性变换 y = x @ W^T + b
        #   - in_features=768：pooler_output 的维度
        #   - out_features=class_cnt：类别数量
        #   - 权重 W 形状：(class_cnt, 768)
        #   - 偏置 b 形状：(class_cnt,)
        #
        # 输入输出形状：
        #   输入:  (batch_size, 768)   → pooler_output
        #   输出:  (batch_size, class_cnt) → logits（未归一化分数）
        # ==================================================================
        in_features = BertConfig.from_pretrained(config.bert_path).hidden_size
        self.linear = nn.Linear(in_features, config.class_cnt)

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        """
        前向传播：生成分类 logits。

        此方法返回原始 logits（未经过 softmax），用于：
        - 计算硬标签损失（CrossEntropyLoss 内部自动做 softmax）
        - 进一步计算软标签（加温度系数 T 后再 softmax）

        参数:
            input_ids (Tensor):       词索引张量，形状 (batch_size, max_length)
                                      [CLS]=101, [SEP]=102, [PAD]=0
            attention_mask (Tensor):  注意力掩码，形状 (batch_size, max_length)
                                      1=有效token，0=填充token
            output_hidden_states (bool): 是否返回每层隐藏状态
                                      True：Transformer 学生做中间层蒸馏时需要
                                      False：BiLSTM 学生做软标签蒸馏时不需要

        返回:
            output_hidden_states=False（默认）：
                Tensor: 分类 logits，形状 (batch_size, class_cnt)
            output_hidden_states=True：
                tuple: (logits, hidden_states)
                    - logits: (batch_size, class_cnt)
                    - hidden_states: tuple of 13 Tensors（含 embedding 层 + 12层）
                      每层形状 (batch, seq_len, 768)
        """
        # ==================================================================
        # 步骤1：BERT 前向传播（冻结参数，不计算梯度）
        #
        # torch.no_grad()：
        #   禁用梯度计算，不构建计算图
        #   效果：节省显存，加速前向传播
        #
        # 为什么这里用 no_grad() 而不是在外部调用时统一加？
        #   - 教师模型在蒸馏时只做推理，参数不更新
        #   - 在 forward 内部加 no_grad() 可以确保 BERT 部分始终不计算梯度
        #   - 但分类头 self.linear 的梯度仍然需要计算（如果教师要微调的话）
        #   - 由于我们已冻结 BERT 参数，no_grad() 主要是为了节省显存
        # ==================================================================
        with torch.no_grad():
            # self.bert_model(input_ids, attention_mask, output_hidden_states)：
            #   output_hidden_states=True 时返回每层 Transformer 的隐藏状态
            #   返回 BaseModelOutputWithPooling 对象
            bert_output = self.bert_model(
                input_ids, attention_mask,
                output_hidden_states=output_hidden_states
            )

        # ==================================================================
        # 步骤2：提取 [CLS] 向量 + 分类
        #
        # 【模型蒸馏知识点：为什么用 pooler_output 而不是 last_hidden_state[:,0]？】
        #
        # pooler_output 的计算过程：
        #   [CLS] 原始向量 (768维) → nn.Linear(768,768) → Tanh
        #   对应源代码：BertModel 的 BertPooler 类
        #
        # 使用 pooler_output 的原因：
        #   1. 语义对齐：pooler_output 是句子级别的语义表示，与下游分类任务对齐
        #   2. 信息丰富：经过 Linear+Tanh 变换，提取了更有区分度的特征
        #   3. 蒸馏适配：pooler_output 经过非线性变换后，概率分布更平滑，
        #      生成的软标签质量更高，更有利于学生模型学习
        #
        # last_hidden_state[:,0] vs pooler_output：
        #   ┌──────────────────────────┬────────────────────────────────────┐
        #   │ last_hidden_state[:,0]    │ 最后一层 Transformer 的 [CLS] 原始输出 │
        #   │                          │ 未经过额外变换，保留原始语义              │
        #   │                          │ 普通分类任务推荐使用                      │
        #   ├──────────────────────────┼────────────────────────────────────┤
        #   │ pooler_output            │ [CLS] → Linear(768,768) → Tanh     │
        #   │                          │ 经过训练优化的句子级表示                 │
        #   │                          │ 蒸馏任务推荐使用（软标签质量更高）         │
        #   └──────────────────────────┴────────────────────────────────────┘
        #
        # 获取 pooler_output 的两种方式：
        #   方式一（推荐）：bert_output.pooler_output
        #   方式二：bert_output[1]（pooler_output 是 BaseModelOutputWithPooling 的第二个属性）
        # ==================================================================
        # 与 04_bert/bert_model.py 保持一致，bert.pkl 的分类头使用该特征训练。
        cls_output = bert_output.last_hidden_state[:, 0]

        # ==================================================================
        # 步骤3：分类头输出 logits
        #
        # self.linear(pooler_output)：
        # 全连接层将 768 维向量映射到 class_cnt 维
        #   输出是原始 logits，未归一化
        #   形状：(batch_size, 768) → (batch_size, class_cnt)
        # ==================================================================
        logits = self.linear(cls_output)

        # 根据 output_hidden_states 决定返回值
        # 中间层蒸馏时，需要教师每层的隐藏状态来监督学生
        if output_hidden_states:
            # bert_output.hidden_states: tuple of 13 Tensors
            #   [0]: embedding 层输出 (batch, seq_len, 768)
            #   [1]~[12]: 12层 Transformer 输出 (batch, seq_len, 768)
            return logits, bert_output.hidden_states
        else:
            return logits

    def get_soft_labels(self, input_ids, attention_mask, temperature=4.0):
        """
        获取带温度系数的软标签（概率分布）。

        【模型蒸馏知识点：温度系数 T 的数学原理】
        
        软标签计算公式：
            q_i = exp(z_i / T) / sum(exp(z_j / T))
        
        其中：
            z_i：教师模型输出的第 i 类 logits
            T：温度系数
            q_i：第 i 类的软标签概率
        
        温度系数 T 的影响：
            T=1（标准 softmax）：
                概率分布尖锐，高概率类别接近 1，其他接近 0
                例：[0.005, 0.002, 0.98, 0.01, 0.003]
                几乎等同于硬标签，暗知识少
            
            T=4（推荐值）：
                概率分布适度平滑，保留类别间的相似性
                例：[0.05, 0.08, 0.65, 0.15, 0.07]
                学生能看到"哪些类别容易混淆"的信息
            
            T=20（高温）：
                概率分布非常平滑，接近均匀分布
                例：[0.18, 0.19, 0.25, 0.20, 0.18]
                暗知识多但噪声也大，可能干扰学生训练
        
        如何选择 T 值？
            - 类别数多（如100类）：T 可以大一些（8~20）
            - 类别数少（如10类）：T 小一些（2~8）
            - 教师模型置信度高：T 大一些，暴露更多暗知识
            - 经验值：T=4 是常用起点，可在 2~10 之间调参

        参数:
            input_ids (Tensor):       词索引张量，形状 (batch_size, max_length)
            attention_mask (Tensor):  注意力掩码，形状 (batch_size, max_length)
            temperature (float):      温度系数 T，默认 4.0
                                      T>1 软化分布，T 越大分布越平滑

        返回:
            Tensor: 软标签概率分布，形状 (batch_size, class_cnt)
                    每行和为 1，值范围 (0, 1)
        """
        # 获取原始 logits
        logits = self.forward(input_ids, attention_mask)

        # 带温度系数的 softmax
        # F.softmax(logits / T, dim=-1)：
        #   - logits / T：用温度系数缩放 logits，T 越大 logits 差异越小
        #   - dim=-1：沿最后一个维度（类别维度）做 softmax
        #   - 返回：概率分布，每行和为 1
        return F.softmax(logits / temperature, dim=-1)


# ==========================================================================
# 测试入口：验证模型结构是否正确
# ==========================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("【知识蒸馏 - 教师模型测试】")
    print("=" * 70)

    # 1- 实例化教师模型
    model = TeacherBertModel()
    print(f"\n模型结构:\n{model}")
    print(f"\n类别数量: {config.class_cnt}")
    print(f"类别列表: {config.class_list}")
    print(f"使用设备: {config.device}")

    # 2- 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"  说明：BERT 参数已冻结，只有分类头的 {trainable_params:,} 个参数可训练")

    # 3- 测试前向传播（模拟数据）
    # 模拟一个 batch=2 的输入
    print("\n" + "=" * 70)
    print("【前向传播测试】")
    print("=" * 70)

    dummy_input_ids = torch.randint(100, 20000, (2, config.max_length))
    dummy_attention_mask = torch.ones(2, config.max_length)

    # 测试普通 logits 输出
    with torch.no_grad():
        logits = model(dummy_input_ids, dummy_attention_mask)
        print(f"Logits 形状: {logits.shape}")  # 期望: (2, class_cnt)

    # 测试软标签输出（不同温度对比）
    print("\n【温度系数对比（第1个样本）】")
    for T in [1, 2, 4, 8, 20]:
        soft_labels = model.get_soft_labels(dummy_input_ids, dummy_attention_mask, temperature=T)
        top3_values, top3_indices = torch.topk(soft_labels[0], k=3)
        print(f"  T={T:2d}: Top-3 类别={top3_indices.tolist()}, 概率={[f'{v:.4f}' for v in top3_values.tolist()]}")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)