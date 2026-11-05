"""
====================================================================================================
test_model.py —— 测试集评估脚本
====================================================================================================
【文件作用】
    加载训练好的学生模型，在测试集上全面评估性能，生成详细报告。
    支持两种学生模型：
    - BiLSTM 学生模型（默认）
    - Transformer 学生模型（通过命令行参数切换）

【使用方式】
    python test_model.py                    # 默认评估 BiLSTM 学生模型
    python test_model.py --model bilstm     # 评估 BiLSTM 学生模型
    python test_model.py --model transformer # 评估 Transformer 学生模型

【输出内容】
    1. 整体指标：准确率、宏平均 F1、精确率、召回率
    2. 各类别指标：每个类别的精确率、召回率、F1
    3. 混淆矩阵（可选）
    4. 教师模型对比（如果有教师模型权重）

【依赖包说明】
----------------------------------------------------------------------------------------------------
1. torch
   - torch.load(path, map_location)       加载模型权重
   - torch.no_grad()                      禁用梯度计算
   - torch.argmax(input, dim)             获取最大值的索引

2. sklearn.metrics（核心评估工具）
   - accuracy_score(y_true, y_pred)       准确率
   - f1_score(y_true, y_pred, average)    F1 值
   - precision_score(y_true, y_pred, average)  精确率
   - recall_score(y_true, y_pred, average)     召回率
   - classification_report(y_true, y_pred)     分类报告（含各类别指标）
   - confusion_matrix(y_true, y_pred)     混淆矩阵

3. data_preprocessing
   - build_dataloader(datapath, shuffle)  构建数据加载器

4. student_bilstm_model / student_transformer_model
   - 学生模型类定义

【面试高频考点 ⭐】
    1. 测试集评估的作用？为什么不能用验证集代替？
    2. 宏平均 vs 微平均的区别？什么时候用哪个？
    3. 如何解读混淆矩阵？对角线和非对角线的含义？
    4. 类别不平衡时哪些指标会失真？如何修正？
    5. 为什么要用 torch.no_grad() 包裹评估代码？
    6. 模型评估时 batch_size 可以更大吗？为什么？
====================================================================================================
"""
import argparse
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from config import Config
from data_preprocessing import build_dataloader
from tqdm import tqdm

# ==========================================================================
# 全局变量
# ==========================================================================
config = Config()


# ==========================================================================
# 函数：load_model
# 作用：根据模型类型加载对应的学生模型
# ==========================================================================
def load_model(model_type='bilstm'):
    """
    加载训练好的模型（支持教师/学生/剪枝模型）。

    参数:
        model_type (str): 模型类型
                          'teacher' / 'prune' / 'bilstm' / 'transformer'

    返回:
        nn.Module: 加载好的模型（已切换到 eval 模式）
    """
    if model_type == 'teacher':
        from teacher_bert_model import TeacherBertModel
        model = TeacherBertModel().to(device=config.device)
        model_path = config.teacher_save_model
    elif model_type == 'prune':
        from teacher_bert_model import TeacherBertModel
        model = TeacherBertModel().to(device=config.device)
        model_path = config.after_prune_path
    elif model_type == 'bilstm':
        from student_bilstm_model import BiLSTMStuModel
        model = BiLSTMStuModel().to(device=config.device)
        model_path = config.student_bilstm_save_model
    elif model_type == 'transformer':
        from student_transformer_model import TransformerStuModel
        model = TransformerStuModel(return_hidden=False).to(device=config.device)
        model_path = config.student_transformer_save_model
    else:
        raise ValueError(f"不支持的模型类型: {model_type}，可选: teacher/prune/bilstm/transformer")

    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型类型: {model_type}")
    print(f"参数量: {total_params:,}")
    print(f"模型路径: {model_path}")

    return model


# ==========================================================================
# 函数：evaluate_test
# 作用：在测试集上全面评估模型性能
# ==========================================================================
def evaluate_test(model, datapath=None):
    """
    在测试集上全面评估模型性能。

    评估流程：
    1. 加载测试集数据
    2. 逐批次推理，收集所有预测结果和真实标签
    3. 计算整体指标（准确率、F1、精确率、召回率）
    4. 生成分类报告（各类别详细指标）
    5. 可选：生成混淆矩阵

    参数:
        model (nn.Module):   加载好的学生模型
        datapath (str|None): 测试集数据路径，None 则使用 config.test_datapath

    返回:
        dict: 评估结果
            {
                'accuracy': float,
                'f1_macro': float,
                'precision_macro': float,
                'recall_macro': float,
                'all_preds': list[int],
                'all_labels': list[int]
            }
    """
    # ------------------------------------------------------------------
    # 1- 加载测试集数据
    #
    # shuffle=False：测试集不需要打乱，保持顺序
    # 如果测试集很大，可以设置更大的 batch_size 加速推理
    # ------------------------------------------------------------------
    if datapath is None:
        datapath = config.test_datapath

    test_loader = build_dataloader(datapath, shuffle=False)
    print(f"测试集路径: {datapath}")
    print(f"测试集批次数: {len(test_loader)}")

    # ------------------------------------------------------------------
    # 2- 逐批次推理
    # ------------------------------------------------------------------
    all_preds = []    # 所有预测标签
    all_labels = []   # 所有真实标签

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(test_loader, desc="测试评估中"):
            # 数据迁移到设备
            input_ids = input_ids.to(device=config.device)
            attention_mask = attention_mask.to(device=config.device)

            # 前向传播
            logits = model(input_ids, attention_mask)
            # 兼容返回 tuple 的模型
            logits = logits[0] if isinstance(logits, tuple) else logits

            # 取预测类别
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            true_labels = labels.cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(true_labels)

    # ------------------------------------------------------------------
    # 3- 计算整体指标
    #
    # 【知识点：评估指标选择】
    # - accuracy：最直观，但类别不平衡时不可靠
    # - F1-macro：各类别 F1 的算术平均，不受类别不平衡影响
    # - 如果类别严重不平衡，macro 比 micro 更能反映小类别的表现
    # ------------------------------------------------------------------
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'all_preds': all_preds,
        'all_labels': all_labels
    }


# ==========================================================================
# 函数：print_classification_report
# 作用：打印详细分类报告
# ==========================================================================
def print_classification_report(all_labels, all_preds):
    """
    打印 sklearn 风格的分类报告，包含每个类别的精确率、召回率、F1。

    【知识点：classification_report 参数】
    - target_names：类别名称列表，用于报告中的标签
    - digits：小数位数
    - zero_division：当某类别无预测时的处理方式

    【知识点：如何解读分类报告？】
    例：类别"体育" precision=0.92, recall=0.85, f1=0.88, support=200
    - precision=0.92：预测为"体育"的样本中，92% 真的是体育
    - recall=0.85：所有真实"体育"样本中，85% 被正确识别
    - f1=0.88：精确率和召回率的调和平均
    - support=200：测试集中有 200 条"体育"样本
    """
    report = classification_report(
        all_labels, all_preds,
        target_names=config.class_list,
        digits=4,
        zero_division=0
    )
    print("\n" + "=" * 60)
    print("【分类报告（各类别详细指标）】")
    print("=" * 60)
    print(report)


# ==========================================================================
# 函数：print_confusion_info
# 作用：打印混淆矩阵的关键信息（易混淆类别对）
# ==========================================================================
def print_confusion_info(all_labels, all_preds):
    """
    打印混淆矩阵中易混淆的类别对（Top-5）。

    帮助发现模型在哪些类别之间容易出错，指导后续优化方向。

    【知识点：混淆矩阵解读】
    混淆矩阵 C[i][j] 表示真实类别 i 被预测为类别 j 的样本数
    - 对角线：正确预测的样本数
    - 非对角线：错误预测的样本数
    - C[i][j] 大的非对角线元素：类别 i 容易和类别 j 混淆
    """
    cm = confusion_matrix(all_labels, all_preds)

    # 找出 Top-5 易混淆的类别对（非对角线最大值）
    errors = []
    for i in range(len(config.class_list)):
        for j in range(len(config.class_list)):
            if i != j and cm[i][j] > 0:
                errors.append((config.class_list[i], config.class_list[j], cm[i][j]))

    errors.sort(key=lambda x: x[2], reverse=True)

    print("\n" + "=" * 60)
    print("【易混淆类别 Top-5（真实类别 → 预测类别）】")
    print("=" * 60)
    for true_cls, pred_cls, count in errors[:5]:
        print(f"  {true_cls} → {pred_cls}: {count} 次")


# ==========================================================================
# 主入口
# ==========================================================================
if __name__ == '__main__':
    # ------------------------------------------------------------------
    # 解析命令行参数
    # --model: 选择模型类型，默认 bilstm
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="知识蒸馏学生模型测试评估")
    parser.add_argument(
        '--model', type=str, default='bilstm',
        choices=['teacher', 'prune', 'bilstm', 'transformer'],
        help='学生模型类型（默认: bilstm）'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("【知识蒸馏学生模型 - 测试集评估】")
    print("=" * 60)
    print(f"设备: {config.device}")
    print(f"类别数量: {config.class_cnt}")
    print(f"类别列表: {config.class_list}")

    # ------------------------------------------------------------------
    # 1- 加载模型
    # ------------------------------------------------------------------
    model = load_model(args.model)

    # ------------------------------------------------------------------
    # 2- 测试集评估
    # ------------------------------------------------------------------
    result = evaluate_test(model)

    # ------------------------------------------------------------------
    # 3- 打印整体指标
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("【整体评估指标】")
    print("=" * 60)
    print(f"  准确率 (Accuracy):    {result['accuracy']:.4f}")
    print(f"  宏平均 F1 (F1-macro):  {result['f1_macro']:.4f}")
    print(f"  宏平均精确率 (Precision): {result['precision_macro']:.4f}")
    print(f"  宏平均召回率 (Recall):    {result['recall_macro']:.4f}")

    # ------------------------------------------------------------------
    # 4- 打印分类报告
    # ------------------------------------------------------------------
    print_classification_report(result['all_labels'], result['all_preds'])

    # ------------------------------------------------------------------
    # 5- 打印混淆信息
    # ------------------------------------------------------------------
    print_confusion_info(result['all_labels'], result['all_preds'])

    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)