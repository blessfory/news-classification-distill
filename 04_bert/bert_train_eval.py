"""
====================================================================================================
bert_train_eval.py —— BERT 文本分类模型训练与评估
====================================================================================================
【文件作用】
    1. 评估函数 eval_model()：在验证集上评估模型性能，计算 F1、准确率、精确率、召回率
    2. 训练函数 train_eval()：完整的训练流程，包含训练+定期评估+模型保存

【依赖包详细说明】
----------------------------------------------------------------------------------------------------
1. torch
   PyTorch 深度学习核心库。
   新增 API：
   - torch.optim                          优化器模块
   - torch.save(obj, path)                保存模型（或 state_dict）到文件
   - torch.load(path)                     从文件加载模型
   - torch.argmax(input, dim)             返回指定维度上最大值的索引
   - tensor.to(device)                    将张量迁移到指定设备（CPU/GPU）
   - tensor.cpu()                         将张量迁移到 CPU
   - tensor.tolist()                      将张量转为 Python 列表
   - tensor.item()                        将标量张量（0维）转为 Python 数值
   - model.state_dict()                   返回模型参数字典（OrderedDict）
   - model.train()                        设置模型为训练模式
   - model.eval()                         设置模型为评估模式

2. torch.nn
   - nn.CrossEntropyLoss(weight=None, reduction='mean', label_smoothing=0.0)
       交叉熵损失函数，结合了 LogSoftmax + NLLLoss
       参数：
         weight:         每个类别的权重 Tensor，用于处理类别不平衡
                         形状 (num_classes,)，默认 None（所有类别权重相同）
         reduction:      聚合方式
                         'mean'：返回批次均值（默认）
                         'sum'：返回批次总和
                         'none'：不聚合，返回每个样本的损失
         label_smoothing: 标签平滑系数（0.0~1.0），减轻过拟合
                         例：0.1 表示将真实标签从 1.0 平滑到 0.9
       注意：输入 logits 不需经过 softmax，CrossEntropyLoss 内部会自动处理

3. torch.optim（优化器模块）
   - torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
       AdamW 优化器：Adam 的改进版，将权重衰减与学习率解耦，效果更好
       参数：
         params:       模型参数迭代器（model.parameters()）
         lr:           学习率（默认 0.001）
         betas:        动量系数（默认 (0.9, 0.999)）
         eps:          数值稳定性常数（默认 1e-8）
         weight_decay: 权重衰减系数（L2 正则化），默认 0.01
                       设为 0 时退化为标准 Adam

   其他常用优化器：
   - torch.optim.Adam(params, lr, ...)        经典 Adam
   - torch.optim.SGD(params, lr, momentum, ...) 随机梯度下降
   - torch.optim.AdamW(params, lr, ...)        解耦权重衰减的 Adam

   优化器固定三步走（每个批次）：
       optim.zero_grad()   清空梯度（PyTorch 默认梯度累加）
       loss.backward()     反向传播，计算梯度
       optim.step()        根据梯度更新参数

4. tqdm
   进度条库，用于显示循环进度。
   常用 API：
   - tqdm(iterable, desc="描述", total=总数)
   - enumerate(tqdm(...), start=1)        带序号的进度条

5. sklearn.metrics（scikit-learn 评估指标模块）
   常用分类指标 API：
   - accuracy_score(y_true, y_pred)        准确率
       公式：(TP + TN) / (TP + TN + FP + FN)
       适用：各类别样本量均衡时

   - precision_score(y_true, y_pred, average)
       精确率（查准率）
       公式：TP / (TP + FP)
       含义：预测为正的样本中，有多少是真正的正类

   - recall_score(y_true, y_pred, average)
       召回率（查全率）
       公式：TP / (TP + FN)
       含义：真正的正类中，有多少被预测出来了

   - f1_score(y_true, y_pred, average)
       F1 值：精确率和召回率的调和平均
       公式：2 * (precision * recall) / (precision + recall)
       范围：0~1，越接近 1 越好

   average 参数（多分类场景）：
     'macro':  先计算每个类别的指标，再求平均（每个类别权重相同）
               适用：样本分布均衡
     'micro':  先汇总所有类别的 TP/FP/FN，再计算指标
               适用：样本分布不均衡
     'weighted': 类似 macro，但按各类别样本数加权平均
     None:      返回每个类别的指标值（列表形式）

   混淆矩阵术语：
     TP (True Positive)：  预测为正，实际为正
     FP (False Positive)： 预测为正，实际为负（误报）
     TN (True Negative)：  预测为负，实际为负
     FN (False Negative)： 预测为负，实际为正（漏报）

【面试高频考点 ⭐】
    1. 训练流程的三步固定操作是什么？（optim.zero_grad() → loss.backward() → optim.step()）
    2. 为什么每次迭代都要 optim.zero_grad()？（PyTorch 默认梯度累加，不清零会叠加）
    3. AdamW 和 Adam 的区别？（AdamW 将权重衰减与学习率解耦，泛化更好）
    4. model.train() 和 model.eval() 的区别？（影响 Dropout 和 BatchNorm）
    5. torch.no_grad() 的作用和原理？（不构建计算图，节省显存，加速推理）
    6. 为什么保存最佳模型而不是最后一个模型？（防止过拟合，选验证集最优）
    7. state_dict 和整个模型保存的区别？（state_dict 更轻量、跨版本兼容性好）
    8. macro/micro/weighted average 的区别和适用场景？
    9. 学习率的选择依据？BERT 微调推荐多少？（2e-5 ~ 5e-5）
    10. 为什么评估时 shuffle=False？（保证每次评估结果一致，方便对比）
    11. CrossEntropyLoss 的 label_smoothing 参数作用？（防止过拟合，标签平滑）
    12. 梯度消失/爆炸如何处理？（梯度裁剪、合适的初始化、BatchNorm）
====================================================================================================
"""
from config import Config
from data_preprocessing import build_dataloader
from bert_model import BertClassifierModel
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# ==========================================================================
# 全局变量
# ==========================================================================
config = Config()


# ==========================================================================
# 函数：eval_model
# 作用：在验证集上评估模型，计算各项分类指标
# ==========================================================================
def eval_model(model, datapath=None):
    """
    评估模型在指定数据集上的性能。

    参数:
        model (nn.Module): 待评估的模型实例
        datapath (str, optional): 数据文件路径。默认 None，使用 config.dev_datapath（验证集）

    返回:
        tuple: (f1score, accuracy, precision, recall)
            - f1score (float):   F1 值（macro 平均）
            - accuracy (float):  准确率
            - precision (float): 精确率（macro 平均）
            - recall (float):    召回率（macro 平均）
    """
    # ------------------------------------------------------------------
    # 步骤1：加载数据
    # 如果未指定 datapath，默认使用验证集（保持向后兼容）
    # shuffle=False（默认）：验证/测试时不打乱数据
    #   原因：训练时 shuffle=True 让模型见更多样化的批次组合，训练更充分
    #         评估时 shuffle=False 保证每次评估结果一致，方便对比
    # ------------------------------------------------------------------
    if datapath is None:
        datapath = config.dev_datapath
    data_loader = build_dataloader(datapath)

    # ------------------------------------------------------------------
    # 步骤2：初始化结果存储变量
    # all_pred_result：收集所有批次的预测标签
    # all_true_result：收集所有批次的真实标签
    # 评估完所有批次后，一次性计算指标
    # ------------------------------------------------------------------
    all_pred_result = []
    all_true_result = []

    # ------------------------------------------------------------------
    # 步骤3：切换模型为评估模式
    #
    # model.eval() vs model.train()：
    #   ┌───────────────┬──────────────────────────────────────────────┐
    #   │ model.train()  │ 训练模式（默认）                              │
    #   │               │ - Dropout 层正常工作（随机丢弃神经元）         │
    #   │               │ - BatchNorm 层使用当前批次的统计量             │
    #   │               │ - 启用梯度计算                                │
    #   ├───────────────┼──────────────────────────────────────────────┤
    #   │ model.eval()   │ 评估模式                                    │
    #   │               │ - Dropout 层停止工作（不丢弃神经元）            │
    #   │               │ - BatchNorm 层使用训练时累积的全局统计量      │
    #   │               │ - 通常配合 torch.no_grad() 使用              │
    #   └───────────────┴──────────────────────────────────────────────┘
    #
    # 重要：如果忘记调用 model.eval()，Dropout 在评估时仍然生效，
    #       会导致每次预测结果不一致，评估指标偏低且不稳定。
    # ------------------------------------------------------------------
    model.eval()

    # ------------------------------------------------------------------
    # 步骤4：遍历验证集，逐批预测
    #
    # torch.no_grad()：
    #   禁用梯度计算，显著节省显存和加速推理
    #   评估/推理时必须使用！
    # ------------------------------------------------------------------
    with torch.no_grad():
        # enumerate(data_loader, start=1)：i 从 1 开始计数，方便日志输出
        for i, (input_ids, attention_mask, labels) in enumerate(tqdm(data_loader), start=1):
            # ----------------------------------------------------------
            # 4.1- 数据迁移到指定设备（GPU/CPU）
            #
            # tensor.to(device)：
            #   将张量从 CPU 复制到 GPU（或反向），返回新张量
            #   参数 device 可以是：
            #     - 字符串：'cpu'、'cuda'、'cuda:0'、'cuda:1'
            #     - torch.device 对象：torch.device('cuda')
            #
            # 注意：模型和数据必须在同一设备上，否则报错
            # ----------------------------------------------------------
            # 从模型参数获取设备，确保模型和数据在同一设备上
            device = next(model.parameters()).device
            input_ids = input_ids.to(device=device)
            attention_mask = attention_mask.to(device=device)
            labels = labels.to(device=device)

            # ----------------------------------------------------------
            # 4.2- 前向传播，获取预测 logits
            # model(input_ids, attention_mask) 等价于 model.forward(...)
            # 推荐写法：model(...)，PyTorch 会自动调用 forward 并处理钩子等
            # ----------------------------------------------------------
            pred_labels = model(input_ids, attention_mask)

            # ----------------------------------------------------------
            # 4.3- 获取预测类别
            #
            # torch.argmax(input, dim)：
            #   返回指定维度上最大值的索引
            #   dim=-1 或 dim=1：沿类别维度取最大值
            #   例：pred_labels = [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]
            #       argmax(dim=-1) = [1, 0]（第0个样本是第1类，第1个样本是第0类）
            # ----------------------------------------------------------
            pred_idx = torch.argmax(pred_labels, dim=-1)

            # ----------------------------------------------------------
            # 4.4- 收集预测结果和真实标签
            #
            # tensor.cpu()：
            #   将 GPU 张量迁移到 CPU（如果已在 CPU 上则无操作）
            #   评估时建议将结果移到 CPU，释放 GPU 显存给训练使用
            #
            # tensor.tolist()：
            #   将张量转为 Python 列表
            #   例：tensor([1, 0, 2]) -> [1, 0, 2]
            #
            # list.extend(iterable)：
            #   将可迭代对象中的元素逐个追加到列表末尾
            #   例：[1, 2].extend([3, 4]) -> [1, 2, 3, 4]
            # ----------------------------------------------------------
            all_pred_result.extend(pred_idx.cpu().tolist())
            all_true_result.extend(labels.cpu().tolist())

    # ------------------------------------------------------------------
    # 步骤5：计算分类评估指标
    #
    # average='macro'：
    #   先计算每个类别的指标，再取平均（每个类别权重相同）
    #   适用场景：样本分布均衡时
    #
    # 如果样本分布不均衡，应使用 average='micro'：
    #   先汇总所有类别的 TP/FP/FN，再计算指标
    #   此时 accuracy = precision = recall = f1 = micro_f1
    #
    # 指标解读：
    #   - accuracy：整体预测正确的比例，不均衡时可能虚高
    #   - precision：预测为某类中真正是该类的比例（误报少→高）
    #   - recall：真实某类中被预测出来的比例（漏报少→高）
    #   - f1：精确率和召回率的调和平均，综合衡量模型性能
    # ------------------------------------------------------------------
    f1score = f1_score(all_true_result, all_pred_result, average='macro')
    precision = precision_score(all_true_result, all_pred_result, average='macro')
    recall = recall_score(all_true_result, all_pred_result, average='macro')
    accuracy = accuracy_score(all_true_result, all_pred_result)

    return f1score, accuracy, precision, recall


# ==========================================================================
# 函数：train_eval
# 作用：完整的训练流程，包含训练循环、定期评估、模型保存
# ==========================================================================
def train_eval():
    """
    训练模型并在验证集上定期评估。

    训练流程：
        1. 加载训练数据
        2. 创建模型、优化器、损失函数
        3. 多轮迭代训练
        4. 每 100 个批次或每轮结束时评估模型
        5. 保存 F1 值最高的模型
    """
    # ==================================================================
    # 步骤1：加载训练数据
    # shuffle=True：训练时打乱数据，让模型每轮看到不同的批次组合
    # ==================================================================
    data_loader = build_dataloader(config.train_datapath, shuffle=True)

    # ==================================================================
    # 步骤2：创建实例对象
    # ==================================================================

    # 2.1- 模型实例化并迁移到指定设备
    #
    # model.to(device)：
    #   将模型的所有参数和缓冲区迁移到指定设备
    #   返回的是同一个模型对象（原地修改），不是复制
    #   必须先加载模型再迁移，否则参数在 CPU 上无法与 GPU 数据运算
    model = BertClassifierModel()
    model = model.to(config.device)

    # 2.2- 优化器实例化
    #
    # torch.optim.AdamW(params, lr, ...)：
    #   AdamW 是 Adam 的改进版，将权重衰减（weight decay）与学习率解耦
    #   相比 Adam，AdamW 的泛化能力通常更好，是 BERT 训练的首选优化器
    #
    # 参数说明：
    #   model.parameters()：模型所有可训练参数
    #   lr=5e-5：学习率，BERT 微调常用范围 2e-5 ~ 5e-5
    #           太大：训练不稳定，可能不收敛
    #           太小：收敛慢，训练时间长
    #
    # 其他常用优化器：
    #   torch.optim.Adam(params, lr=1e-3)   通用 Adam
    #   torch.optim.SGD(params, lr=1e-2)    随机梯度下降
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 2.3- 损失函数实例化
    #
    # nn.CrossEntropyLoss()：
    #   多分类交叉熵损失
    #   内部计算：先对 logits 做 softmax 归一化，再计算负对数似然
    #   输入：pred_labels（原始 logits，不需 softmax）和 labels（类别索引，非 one-hot）
    #
    # 公式：Loss = -log(exp(x[class]) / sum(exp(x)))
    #
    # 常用参数：
    #   weight=None：类别权重，处理类别不平衡
    #   reduction='mean'：返回批次平均值
    criterion = nn.CrossEntropyLoss()

    # ==================================================================
    # 步骤3：切换模型为训练模式
    #
    # model.train()：
    #   启用 Dropout（随机丢弃神经元）和 BatchNorm 的批次统计
    #   训练时必须在 model.train() 状态下进行
    # ==================================================================
    model.train()

    # ==================================================================
    # 步骤4：多轮训练循环
    # ==================================================================
    epochs = 5
    best_f1_score = 0.0  # 记录历史最高 F1 值，用于模型保存

    for epoch in range(epochs):
        # total_loss：累积本轮所有批次的损失值，用于计算平均损失
        total_loss = 0.0

        # enumerate(..., start=1)：i 从 1 开始，方便打印 "第i批次"
        for i, (input_ids, attention_mask, labels) in enumerate(tqdm(data_loader), start=1):

            # ----------------------------------------------------------
            # 4.1- 数据迁移到指定设备（GPU/CPU）
            # ----------------------------------------------------------
            input_ids = input_ids.to(device=config.device)
            attention_mask = attention_mask.to(device=config.device)
            labels = labels.to(device=config.device)

            # ----------------------------------------------------------
            # 4.2- 前向传播：计算预测值
            # model(input_ids, attention_mask) 自动调用 forward 方法
            # ----------------------------------------------------------
            pred_labels = model(input_ids, attention_mask)

            # ----------------------------------------------------------
            # 4.3- 计算损失值
            #
            # criterion(pred, target)：
            #   pred：模型输出 logits，形状 (batch, class_cnt)
            #   target：真实标签，形状 (batch,)，每个元素是类别索引
            #   返回：标量损失值（reduction='mean' 时）
            #
            # loss.item()：
            #   将标量张量转为 Python 浮点数
            #   用于累加时避免保留计算图，节省显存
            # ----------------------------------------------------------
            loss = criterion(pred_labels, labels)
            total_loss += loss.item()

            # ----------------------------------------------------------
            # 4.4- 反向传播与参数更新（固定三步）
            #
            # 步骤① optim.zero_grad()：
            #   清空所有参数的梯度缓存
            #   PyTorch 默认会累加梯度（方便 RNN 等场景），所以每步必须清零
            #   如果不清零，梯度会累加，导致参数更新错误
            #
            # 步骤② loss.backward()：
            #   反向传播，计算所有 requires_grad=True 的参数的梯度
            #   梯度存储在 param.grad 中
            #
            # 步骤③ optim.step()：
            #   根据梯度更新所有参数
            #   更新公式（AdamW）：θ = θ - lr * (m_hat / (sqrt(v_hat) + eps) + wd * θ)
            # ----------------------------------------------------------
            optim.zero_grad()
            loss.backward()
            optim.step()

            # ----------------------------------------------------------
            # 4.5- 定期评估：每 100 个批次或最后一个批次时评估
            #
            # 评估频率权衡：
            #   太频繁（如每 10 步）：拖慢训练速度
            #   太稀疏（如每轮结束）：可能错过最佳模型
            #   推荐：每 100~500 步或每半个 epoch 评估一次
            # ----------------------------------------------------------
            if i % 100 == 0 or i == len(data_loader):
                # 4.5.1- 调用评估函数，获取各项指标
                f1score, accuracy, precision, recall = eval_model(model)

                # 计算当前批次的平均损失
                avg_loss = total_loss / i

                print(
                    f"第{epoch + 1}轮次，第{i}批次，"
                    f"平均损失={avg_loss:.4f}，"
                    f"F1值={f1score:.4f}，"
                    f"准确率={accuracy:.4f}，"
                    f"精确率={precision:.4f}，"
                    f"召回率={recall:.4f}"
                )

                # 4.5.2- 模型保存（仅保存最佳模型）
                #
                # torch.save(obj, path)：
                #   保存对象到文件
                #   保存 state_dict 而非整个模型：
                #     model.state_dict()：返回 OrderedDict，包含所有参数名称和值
                #     优点：只保存参数，文件更小，加载更灵活（不依赖模型类定义）
                #
                # 保存整个模型（不推荐）：
                #   torch.save(model, path)  → 文件大，依赖类定义，跨版本兼容性差
                #
                # 加载方式：
                #   model.load_state_dict(torch.load(path))
                if f1score >= best_f1_score:
                    torch.save(model.state_dict(), config.save_model)
                    best_f1_score = f1score
                    print(f"  >>> 模型已保存！当前最佳 F1={best_f1_score:.4f}")

                # 4.5.3- 评估后切回训练模式
                # eval_model() 中调用了 model.eval()，需要切回来
                # 如果不切回，Dropout 失效，后续训练效果会变差
                model.train()

    # ==================================================================
    # 训练完成
    # ==================================================================
    print("=" * 60)
    print(f"训练完成！最佳 F1 值：{best_f1_score:.4f}")
    print(f"模型保存路径：{config.save_model}")
    print("=" * 60)


# ==========================================================================
# 程序入口
# ==========================================================================
if __name__ == '__main__':
    train_eval()