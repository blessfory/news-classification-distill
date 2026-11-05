"""
====================================================================================================
model_distll.py —— 知识蒸馏训练主脚本
====================================================================================================
【文件作用】
    1. eval_model()：评估模型在验证集上的性能（F1、准确率、精确率、召回率）
    2. train_model()：知识蒸馏训练流程，用教师模型（BERT）的软标签指导学生模型学习

【知识蒸馏核心流程】
    第1步：加载预训练好的教师模型（BERT，参数冻结）
    第2步：创建学生模型（BiLSTM 或 3层 Transformer，从头训练）
    第3步：每个 batch 同时计算：
          - 硬标签损失：学生输出 vs 真实标签（CrossEntropyLoss）
          - 软标签损失：学生输出 vs 教师输出（KL 散度，带温度 T）
    第4步：总损失 = (1-α) × 硬标签损失 + α × T² × 软标签损失
    第5步：反向传播，更新学生模型参数
    第6步：每轮评估，保存最佳模型
====================================================================================================
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from config import Config
from data_preprocessing import build_dataloader
from teacher_bert_model import TeacherBertModel
from student_bilstm_model import BiLSTMStuModel
from student_transformer_model import TransformerStuModel

# ==========================================================================
# 全局变量
# ==========================================================================
config = Config()

# 确保 save_model 目录存在
os.makedirs(os.path.dirname(config.student_bilstm_save_model), exist_ok=True)


# ==========================================================================
# 函数：eval_model
# 作用：评估模型在数据集上的性能（兼容教师模型和学生模型）
# ==========================================================================
def eval_model(model, datapath=None):
    """
    评估模型在验证集/测试集上的性能。

    参数:
        model (nn.Module):    模型实例（教师或学生都可以）
        datapath (str|None):  数据路径，None 则使用 config.dev_datapath

    返回:
        tuple: (f1score, accuracy, precision, recall)
    """
    if datapath is None:
        datapath = config.dev_datapath

    valid_loader = build_dataloader(datapath, shuffle=False)
    model.eval()

    all_pred_result = []
    all_true_result = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(valid_loader, desc="评估中"):
            input_ids = input_ids.to(device=config.device)
            attention_mask = attention_mask.to(device=config.device)

            output = model(input_ids, attention_mask)
            # 兼容返回 tuple 的模型（如 TransformerStuModel(return_hidden=True)）
            logits = output[0] if isinstance(output, tuple) else output
            pred_labels = torch.argmax(logits, dim=-1).cpu().tolist()
            true_labels = labels.cpu().tolist()

            all_pred_result.extend(pred_labels)
            all_true_result.extend(true_labels)

    f1score = f1_score(all_true_result, all_pred_result, average='macro')
    precision = precision_score(all_true_result, all_pred_result, average='macro', zero_division=0)
    recall = recall_score(all_true_result, all_pred_result, average='macro', zero_division=0)
    accuracy = accuracy_score(all_true_result, all_pred_result)

    return f1score, accuracy, precision, recall


# ==========================================================================
# 函数：train_model
# 作用：知识蒸馏训练主流程
# ==========================================================================
def train_model(model_type='bilstm'):
    """
    知识蒸馏训练主流程。

    参数:
        model_type (str): 学生模型类型
                          'bilstm'     - BiLSTM 学生模型（约 9.6M 参数，推理最快）
                          'transformer' - 3层 Transformer 学生模型（约 15M 参数，支持中间层蒸馏）
    """
    # ==================================================================
    # 1- 加载数据
    # ==================================================================
    train_loader = build_dataloader(config.train_datapath, shuffle=True)

    # ==================================================================
    # 2- 创建模型
    # ==================================================================

    # 2.1- 加载训练好的教师模型（BERT）
    print(f"\n加载教师模型: {config.teacher_save_model}")
    teacher_model = TeacherBertModel().to(device=config.device)
    teacher_model.load_state_dict(
        torch.load(config.teacher_save_model, map_location=config.device)
    )
    teacher_model.eval()
    print(f"教师模型加载完成，设备: {config.device}")

    # 2.2- 创建学生模型
    if model_type == 'bilstm':
        student_model = BiLSTMStuModel().to(device=config.device)
        use_hidden_distill = False
        student_save_model = config.student_bilstm_save_model
    elif model_type == 'transformer':
        student_model = TransformerStuModel(return_hidden=True).to(device=config.device)
        use_hidden_distill = True
        student_save_model = config.student_transformer_save_model
    else:
        raise ValueError(f"不支持的学生模型类型: {model_type}")

    total_params = sum(p.numel() for p in student_model.parameters())
    print(f"学生模型参数量: {total_params:,}")

    # ==================================================================
    # 3- 损失函数、优化器、学习率调度器
    # ==================================================================
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(student_model.parameters(), lr=1e-4)

    epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=epochs, eta_min=1e-6
    )

    # ==================================================================
    # 4- 蒸馏超参数
    #
    # 【知识点】
    #   T（温度系数）：控制软标签的平滑程度，T 越大分布越平滑，暗知识越明显
    #   α（alpha）：软标签损失的权重，α 越大越依赖教师模型
    #   软标签损失乘 T²：补偿温度缩放导致的梯度缩小
    # ==================================================================
    T = 2          # 温度系数
    alpha = 0.7    # 软标签损失权重

    # ==================================================================
    # 5- 训练循环
    # ==================================================================
    best_f1 = 0.0

    for epoch in range(epochs):
        current_lr = optim.param_groups[0]['lr']
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}  学习率: {current_lr:.2e}")
        print(f"{'='*60}")

        student_model.train()
        total_loss = 0.0
        total_hard_loss = 0.0
        total_soft_loss = 0.0

        for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = input_ids.to(device=config.device)
            attention_mask = attention_mask.to(device=config.device)
            labels = labels.to(device=config.device)

            # 教师模型生成软标签（无梯度）
            with torch.no_grad():
                if use_hidden_distill:
                    teacher_logits, teacher_hiddens = teacher_model(
                        input_ids, attention_mask, output_hidden_states=True
                    )
                else:
                    teacher_logits = teacher_model(input_ids, attention_mask)
                teacher_soft = F.softmax(teacher_logits / T, dim=-1)

            # 学生模型前向传播
            if use_hidden_distill:
                student_logits, student_hiddens = student_model(
                    input_ids, attention_mask
                )
            else:
                student_logits = student_model(input_ids, attention_mask)

            # 硬标签损失
            hard_loss = criterion(student_logits, labels)

            # 软标签损失（KL 散度）
            student_log_soft = F.log_softmax(student_logits / T, dim=-1)
            soft_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean') * (T * T)

            # 中间层蒸馏损失（仅 Transformer 学生）
            if use_hidden_distill:
                hidden_loss = 0.0
                # 学生 3 层 → 教师 12 层，每 4 层对齐
                distill_layers = {0: 3, 1: 7, 2: 11}
                for stu_idx, tea_idx in distill_layers.items():
                    tea_hidden = teacher_hiddens[tea_idx + 1]  # +1 跳过 embedding 层
                    stu_hidden = student_hiddens[stu_idx]
                    hidden_loss += F.mse_loss(stu_hidden, tea_hidden)
                loss = (1 - alpha) * hard_loss + alpha * 0.5 * soft_loss + alpha * 0.5 * hidden_loss
            else:
                loss = (1 - alpha) * hard_loss + alpha * soft_loss

            # 反向传播
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optim.step()

            total_loss += loss.item()
            total_hard_loss += hard_loss.item()
            total_soft_loss += soft_loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        avg_hard = total_hard_loss / len(train_loader)
        avg_soft = total_soft_loss / len(train_loader)
        print(f"平均损失: {avg_loss:.4f}  硬标签: {avg_hard:.4f}  软标签: {avg_soft:.4f}")

        # 评估
        f1score, accuracy, precision, recall = eval_model(student_model)
        print(f"验证集: F1={f1score:.4f}  Acc={accuracy:.4f}  Prec={precision:.4f}  Rec={recall:.4f}")

        if f1score > best_f1:
            best_f1 = f1score
            torch.save(student_model.state_dict(), student_save_model)
            print(f"✅ 最佳模型已保存，F1={best_f1:.4f}")

    print(f"\n训练完成！最佳 F1={best_f1:.4f}")
    return best_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练知识蒸馏学生模型')
    parser.add_argument(
        '--model',
        choices=['bilstm', 'transformer'],
        default='bilstm',
        help='学生模型类型，默认 bilstm',
    )
    args = parser.parse_args(args=["--model", 'transformer'])

    print("=" * 60)
    print("知识蒸馏训练")
    print("=" * 60)
    print(f"教师模型: {config.teacher_save_model}")
    print(f"BiLSTM 学生模型: {config.student_bilstm_save_model}")
    print(f"Transformer 学生模型: {config.student_transformer_save_model}")
    print(f"设备: {config.device}")
    print(f"类别数: {config.class_cnt}")
    train_model(model_type=args.model)