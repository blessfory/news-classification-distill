"""
====================================================================================================
bert_predict.py —— BERT 模型测试集评估与预测结果导出
====================================================================================================
【文件作用】
    加载训练好的 BERT 模型，在测试集（test.txt）上进行最终评估，
    输出分类指标（F1/准确率/精确率/召回率），并将预测结果保存到 CSV 文件。

【为什么需要测试集评估】
    标准 ML 流程中，数据分为三部分：
    ┌──────────┬──────────────────────────────────────────────────┐
    │ 训练集     │ 用于训练模型参数                                 │
    │ train.txt │                                                  │
    ├──────────┼──────────────────────────────────────────────────┤
    │ 验证集     │ 用于调参、选模型、早停                             │
    │ dev.txt   │ 训练过程中反复使用，可能导致"间接过拟合"            │
    ├──────────┼──────────────────────────────────────────────────┤
    │ 测试集     │ 只在最后使用一次，评估模型的真实泛化能力             │
    │ test.txt  │ 最重要的指标来源！                                │
    └──────────┴──────────────────────────────────────────────────┘

    ⭐ 面试考点：为什么测试集只能使用一次？什么是数据泄露（Data Leakage）？

【与 bert_train_eval.py 的关系】
    - bert_train_eval.py：训练时在验证集上评估，用于监控训练过程
    - bert_predict.py：训练完成后在测试集上评估，用于最终报告

【依赖包说明】
    1. pandas
       - pd.DataFrame()  创建表格数据结构
       - df.to_csv()     导出为 CSV 文件
       - 常用参数：index=False（不导出行索引）, encoding='utf-8-sig'（兼容 Excel 打开）

    2. torch
       - torch.load(path, map_location)  加载模型权重
       - model.load_state_dict(state_dict)  将权重赋值给模型
       - model.eval()  切换评估模式

    3. 本项目其他模块
       - config.py:              全局配置
       - bert_model.py:          BERT 分类模型
       - bert_train_eval.py:     评估函数 eval_model()
       - data_preprocessing.py:  build_dataloader() 数据加载
====================================================================================================
"""
import pandas as pd
import torch
from tqdm import tqdm

from config import Config
from bert_model import BertClassifierModel
from bert_train_eval import eval_model, build_dataloader


# ==========================================================================
# 程序入口
# ==========================================================================
if __name__ == '__main__':
    # ======================================================================
    # 步骤1：加载配置
    # ======================================================================
    print("=" * 60)
    print("【步骤1】加载配置")
    print("=" * 60)
    config = Config()
    print(f"运算设备：{config.device}")
    print(f"测试集路径：{config.test_datapath}")
    print(f"模型路径：{config.save_model}")
    print(f"类别数量：{config.class_cnt}")
    print(f"类别列表：{config.class_list}")

    # ======================================================================
    # 步骤2：加载训练好的模型
    #
    # 流程：
    #   2.1- 创建与训练时结构完全一致的模型实例
    #   2.2- 加载训练好的模型权重（state_dict）
    #   2.3- 将模型迁移到指定设备（GPU/CPU）
    #   2.4- 切换到评估模式（禁用 Dropout、BatchNorm 训练行为）
    #
    # torch.load() 参数详解：
    #   map_location: 指定模型权重加载到哪个设备
    #     - 'cpu'：强制加载到 CPU（GPU 训练的模型可用于 CPU 推理）
    #     - 'cuda'：加载到默认 GPU
    #     - config.device：自动适配（有 GPU 用 GPU，无 GPU 用 CPU）
    #
    # ⭐ 面试考点：
    #   - torch.load 和 model.load_state_dict 的区别？
    #   - map_location 参数的作用？什么场景下必须用？
    #   - 为什么加载权重后要调用 model.eval()？
    # ======================================================================
    print("\n" + "=" * 60)
    print("【步骤2】加载模型")
    print("=" * 60)

    model = BertClassifierModel()
    model.load_state_dict(torch.load(config.save_model, map_location=config.device))
    model.to(config.device)
    model.eval()
    print(f"模型已加载到设备：{next(model.parameters()).device}")

    # ======================================================================
    # 步骤3：在测试集上评估模型
    #
    # eval_model() 内部流程：
    #   1. 加载指定数据集（测试集）
    #   2. 逐批前向传播，收集预测结果
    #   3. 关闭梯度计算（torch.no_grad()）
    #   4. 计算 F1/准确率/精确率/召回率（macro 平均）
    #
    # macro 平均 vs micro 平均：
    #   - macro：每个类别单独计算指标，然后取平均（不受类别不平衡影响）
    #   - micro：将所有类别的 TP/FP/FN 加总后计算（受大类别影响大）
    #   - 多分类通常用 macro，更公平
    #
    # ⭐ 面试考点：
    #   - macro 平均和 micro 平均的区别？什么时候用哪个？
    #   - 为什么测试集评估要关闭梯度计算？
    # ======================================================================
    print("\n" + "=" * 60)
    print("【步骤3】测试集评估")
    print("=" * 60)

    f1, acc, prec, rec = eval_model(model, datapath=config.test_datapath)

    print(f"\n测试集最终指标：")
    print(f"  F1 值    = {f1:.4f}")
    print(f"  准确率    = {acc:.4f}")
    print(f"  精确率    = {prec:.4f}")
    print(f"  召回率    = {rec:.4f}")

    # ======================================================================
    # 步骤4：导出预测结果到 CSV
    #
    # 目的：方便后续做错误分析（Bad Case Analysis），
    #       找出模型在哪些类别上容易混淆，针对性地优化。
    #
    # CSV 列说明：
    #   title：      原始新闻标题
    #   true_label： 真实类别（标签）
    #   pred_label： 模型预测类别
    #   is_correct： 是否预测正确（True/False）
    #
    # pandas 常用操作：
    #   df.to_csv(path, index=False, encoding='utf-8-sig')
    #     - index=False：不保存行索引（0,1,2...）
    #     - encoding='utf-8-sig'：带 BOM 的 UTF-8，Excel 打开不乱码
    #
    # ⭐ 面试考点：
    #   - 如何做错误分析（Bad Case Analysis）？
    #   - 预测结果保存为 CSV 有什么好处？（可追溯、可分析、可可视化）
    # ======================================================================
    print("\n" + "=" * 60)
    print("【步骤4】导出预测结果到 CSV")
    print("=" * 60)

    data_loader = build_dataloader(config.test_datapath)

    all_titles = []
    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(data_loader, desc="导出预测结果"):
            device = next(model.parameters()).device
            input_ids = input_ids.to(device=device)
            attention_mask = attention_mask.to(device=device)

            pred_labels = model(input_ids, attention_mask)
            pred_idx = torch.argmax(pred_labels, dim=-1)

            all_pred_labels.extend(pred_idx.cpu().tolist())
            all_true_labels.extend(labels.tolist())

    # 将数字标签映射为类别名称
    # config.class_list 格式：['体育', '娱乐', '科技', ...]
    pred_names = [config.class_list[i] for i in all_pred_labels]
    true_names = [config.class_list[i] for i in all_true_labels]

    # 读取测试集原始文本（用于 CSV 输出）
    titles = []
    with open(config.test_datapath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                # 测试集格式：标题\t类别编号
                titles.append(parts[0])

    # 确保标题数量与预测数量一致（防止数据加载时过滤了某些行）
    min_len = min(len(titles), len(pred_names))
    titles = titles[:min_len]
    pred_names = pred_names[:min_len]
    true_names = true_names[:min_len]

    # 构建 DataFrame 并导出
    df = pd.DataFrame({
        'title': titles,
        'true_label': true_names,
        'pred_label': pred_names,
        'is_correct': [p == t for p, t in zip(pred_names, true_names)]
    })

    output_path = config.save_model.replace('.pkl', '_test_predictions.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"预测结果已保存到：{output_path}")
    print(f"总样本数：{len(df)}")
    print(f"正确数：{df['is_correct'].sum()}")
    print(f"错误数：{(~df['is_correct']).sum()}")

    # 打印每个类别的准确率（方便发现类别偏差）
    print("\n各类别准确率：")
    for cls_name in config.class_list:
        cls_df = df[df['true_label'] == cls_name]
        if len(cls_df) > 0:
            cls_acc = cls_df['is_correct'].mean()
            print(f"  {cls_name}: {cls_acc:.4f} ({cls_df['is_correct'].sum()}/{len(cls_df)})")

    print("\nBERT 测试集评估完成！")