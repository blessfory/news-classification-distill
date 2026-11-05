"""
====================================================================================================
prune.py —— 非结构化 L1 剪枝脚本
====================================================================================================
【文件作用】
    对训练好的 BERT 教师模型进行非结构化 L1 剪枝，为后续蒸馏做准备。
    流程：剪枝前评估 → 剪枝 → 剪枝后评估 → 微调恢复 → 保存

【剪枝核心概念】
    1. 非结构化剪枝：按权重绝对值大小，将最小的 individual 权重置零
       - 优点：精度损失小，粒度细
       - 缺点：不改变矩阵形状，需硬件/框架支持稀疏计算才能加速
    2. L1 剪枝原则：|w| 越小 → 对输出的贡献越小 → 可以安全剪掉
    3. 全局剪枝：所有层的 Q/K/V 权重一起排序，重要的层自动保留更多权重

【剪枝策略】
    剪枝目标：Q（query）、K（key）、V（value）的权重矩阵
    剪枝比例：10%（可调整）
    方法：prune.L1Unstructured + prune.global_unstructured

【与蒸馏的关系】
    先剪枝 → 后蒸馏：
    剪枝会损失精度，蒸馏用大模型知识修复精度损失
====================================================================================================
"""
import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from config import Config
from data_preprocessing import build_dataloader
from teacher_bert_model import TeacherBertModel

# ==========================================================================
# 全局变量
# ==========================================================================
config = Config()
os.makedirs(os.path.dirname(config.after_prune_path), exist_ok=True)


# ==========================================================================
# 函数：eval_model_inline
# 作用：自包含的模型评估（不依赖 model_distll.py）
# ==========================================================================
def eval_model_inline(model, datapath=None):
    """
    评估模型性能（自包含版本，不依赖 model_distll.py）。

    参数:
        model (nn.Module):    模型实例
        datapath (str|None):  数据路径，None 则用 config.dev_datapath

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

            logits = model(input_ids, attention_mask)
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
# 函数：compute_sparsity
# 作用：统计模型权重中值为 0 的比例
# ==========================================================================
def compute_sparsity(model, verbose=True):
    """
    计算模型 Q/K/V 权重的稀疏度（值为 0 的比例）。

    参数:
        model (nn.Module): 模型实例
        verbose (bool):    是否逐层打印

    返回:
        float: 全局稀疏度（0~1），表示值为 0 的权重占比
    """
    total_zeros = 0
    total_params = 0
    num_layers = model.bert_model.config.num_hidden_layers

    if verbose:
        print(f"\n{'='*60}")
        print("稀疏度统计（Q/K/V 权重）")
        print(f"{'='*60}")

    for layer_idx in range(num_layers):
        for proj_name in ['query', 'key', 'value']:
            try:
                weight = getattr(
                    model.bert_model.encoder.layer[layer_idx].attention.self,
                    proj_name
                ).weight
                zero_cnt = (weight == 0).sum().item()
                total_cnt = weight.numel()
                total_zeros += zero_cnt
                total_params += total_cnt

                if verbose:
                    sparsity = zero_cnt / total_cnt * 100
                    print(f"  Layer {layer_idx:2d} {proj_name:5s}: {sparsity:5.1f}% 零值 ({zero_cnt}/{total_cnt})")
            except AttributeError:
                continue

    if total_params == 0:
        return 0.0

    overall_sparsity = total_zeros / total_params
    if verbose:
        print(f"\n  全局稀疏度: {overall_sparsity*100:.2f}% ({total_zeros}/{total_params})")

    return overall_sparsity


# ==========================================================================
# 函数：build_prune_parameters
# 作用：构建剪枝参数列表
# ==========================================================================
def build_prune_parameters(model, num_layers=None):
    """
    构建需要剪枝的参数列表（Q/K/V 权重）。

    参数:
        model (nn.Module):     模型实例
        num_layers (int|None): 剪枝的层数，None 表示所有层

    返回:
        list[tuple]: [(module, 'weight'), ...]
    """
    if num_layers is None:
        num_layers = model.bert_model.config.num_hidden_layers

    parameters_to_prune = []
    for layer_idx in range(num_layers):
        multi_self_att = model.bert_model.encoder.layer[layer_idx].attention.self
        for proj_name in ['query', 'key', 'value']:
            parameters_to_prune.append(
                (getattr(multi_self_att, proj_name), 'weight')
            )

    print(f"剪枝目标: {len(parameters_to_prune)} 个参数矩阵（{num_layers} 层 × 3 个权重）")
    return parameters_to_prune


# ==========================================================================
# 函数：ensure_contiguous
# 作用：确保所有权重张量在内存中连续存储
# ==========================================================================
def ensure_contiguous(parameters_to_prune):
    """
    确保所有权重张量连续存储，避免 view/stride 报错。

    参数:
        parameters_to_prune (list): [(module, 'weight'), ...]
    """
    for module, param_name in parameters_to_prune:
        param = getattr(module, param_name)
        if not param.is_contiguous():
            param.data = param.data.contiguous()


# ==========================================================================
# 函数：remove_pruning_masks
# 作用：固化剪枝掩码，使剪枝永久生效
# ==========================================================================
def remove_pruning_masks(model, num_layers=None):
    """
    移除剪枝的掩码包装，将剪枝永久固化到权重中。

    参数:
        model (nn.Module):     模型实例
        num_layers (int|None): 层数
    """
    if num_layers is None:
        num_layers = model.bert_model.config.num_hidden_layers

    print(f"\n固化剪枝掩码...")
    for layer_idx in range(num_layers):
        multi_self_att = model.bert_model.encoder.layer[layer_idx].attention.self
        for proj_name in ['query', 'key', 'value']:
            try:
                prune.remove(getattr(multi_self_att, proj_name), 'weight')
            except Exception:
                pass
    print("掩码固化完成")


# ==========================================================================
# 主流程
# ==========================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("【BERT 模型剪枝流程】")
    print("=" * 70)
    print(f"教师模型路径: {config.teacher_save_model}")
    print(f"剪枝后保存路径: {config.after_prune_path}")
    print(f"设备: {config.device}")
    print(f"类别数: {config.class_cnt}")

    # ==================================================================
    # 步骤1：加载模型
    # ==================================================================
    print(f"\n{'='*70}")
    print("步骤1：加载教师模型")
    print(f"{'='*70}")

    model = TeacherBertModel()
    state_dict = torch.load(config.teacher_save_model, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=config.device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print(f"模型设备: {next(model.parameters()).device}")

    # ==================================================================
    # 步骤2：剪枝前评估
    # ==================================================================
    print(f"\n{'='*70}")
    print("步骤2：剪枝前评估")
    print(f"{'='*70}")

    f1, acc, prec, rec = eval_model_inline(model)
    print(f"剪枝前  →  F1={f1:.4f}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

    # ==================================================================
    # 步骤3：执行剪枝
    # ==================================================================
    print(f"\n{'='*70}")
    print("步骤3：执行 L1 非结构化剪枝")
    print(f"{'='*70}")

    prune_amount = 0.1  # 剪掉 10% 的权重

    # 3.1- 构建剪枝参数列表
    parameters_to_prune = build_prune_parameters(model)

    # 3.2- 确保内存连续（修复 view/stride 报错）
    ensure_contiguous(parameters_to_prune)

    # 3.3- 全局非结构化 L1 剪枝
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_amount,
    )
    print(f"剪枝完成！剪掉了 {prune_amount*100:.0f}% 的 Q/K/V 权重")

    # 3.4- 固化剪枝掩码
    remove_pruning_masks(model)

    # ==================================================================
    # 步骤4：剪枝后评估
    # ==================================================================
    print(f"\n{'='*70}")
    print("步骤4：剪枝后评估")
    print(f"{'='*70}")

    # 统计稀疏度
    sparsity = compute_sparsity(model)

    # 评估
    f1_after, acc_after, prec_after, rec_after = eval_model_inline(model)

    # ==================================================================
    # 步骤5：效果对比
    # ==================================================================
    print(f"\n{'='*70}")
    print("步骤5：剪枝效果对比")
    print(f"{'='*70}")
    print(f"{'指标':<12} {'剪枝前':>10} {'剪枝后':>10} {'变化':>10}")
    print(f"{'-'*42}")
    print(f"{'F1-score':<12} {f1:>10.4f} {f1_after:>10.4f} {f1_after-f1:>+10.4f}")
    print(f"{'Accuracy':<12} {acc:>10.4f} {acc_after:>10.4f} {acc_after-acc:>+10.4f}")
    print(f"{'Precision':<12} {prec:>10.4f} {prec_after:>10.4f} {prec_after-prec:>+10.4f}")
    print(f"{'Recall':<12} {rec:>10.4f} {rec_after:>10.4f} {rec_after-rec:>+10.4f}")
    print(f"{'稀疏度':<12} {'0.00%':>10} {sparsity*100:>9.2f}% {'--':>10}")

    # ==================================================================
    # 步骤6：保存剪枝后模型
    # ==================================================================
    torch.save(model.state_dict(), config.after_prune_path)
    print(f"\n剪枝后模型已保存至: {config.after_prune_path}")

    print(f"\n{'='*70}")
    print("剪枝流程完成！")
    print(f"下一步：用剪枝后模型作为教师，进行知识蒸馏训练")
    print(f"  python model_distll.py")
    print(f"{'='*70}")