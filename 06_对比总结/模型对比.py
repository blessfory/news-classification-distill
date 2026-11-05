"""
====================================================================================================
模型对比.py —— 全模型效果对比脚本（含蒸馏模型）
====================================================================================================
【文件作用】
    在测试集上统一评估所有模型，包括传统 ML、BERT 及其蒸馏/剪枝/量化版本，
    输出对比表格，包括分类指标和推理速度。

【模型总览】
    ┌────────────────┬──────────────┬──────────────────┬──────────────────────┐
    │ 模型            │ 特征提取      │ 分类器             │ 技术特点              │
    ├────────────────┼──────────────┼──────────────────┼──────────────────────┤
    │ 随机森林 (RF)    │ TF-IDF 稀疏   │ RandomForest     │ 传统 ML，可解释性强    │
    │ XGBoost         │ Word2Vec 密集 │ XGBoost          │ 梯度提升，效果通常最好   │
    │ FastText        │ 字符级 n-gram │ 浅层神经网络       │ 工业级，极快训练速度    │
    │ BERT            │ Transformer   │ 预训练 + 微调      │ 深度语义理解，效果最好   │
    ├────────────────┼──────────────┼──────────────────┼──────────────────────┤
    │ BERT-DQ (量化)   │ Transformer   │ 动态量化 (int8)    │ 体积缩小 4 倍，精度略降  │
    │ BERT-Prune (剪枝)│ Transformer   │ 非结构化 L1 剪枝   │ 10% 权重置零，稀疏化     │
    │ BiLSTM-蒸馏      │ BiLSTM       │ 知识蒸馏           │ 9.6M 参数，轻量        │
    │ Transformer-蒸馏  │ 3层Transformer│ 知识蒸馏           │ 1.5M 参数，Transformer  │
    └────────────────┴──────────────┴──────────────────┴──────────────────────┘

【评估指标说明】
    - F1 值：精确率和召回率的调和平均，F1 = 2*P*R/(P+R)
    - 准确率：预测正确的样本数 / 总样本数
    - 精确率：预测为正类中真正为正类的比例
    - 召回率：真正为正类中被正确预测的比例
    - 推理速度：处理单条样本的平均耗时（毫秒）

【依赖包说明】
    1. importlib.util
       - importlib.util.spec_from_file_location()  从文件路径加载模块
       - 用途：避免不同目录下同名模块（config.py）的导入冲突

    2. sklearn.metrics
       - f1_score(y_true, y_pred, average='macro')
       - accuracy_score(y_true, y_pred)
       - precision_score(y_true, y_pred, average='macro')
       - recall_score(y_true, y_pred, average='macro')

    3. pandas
       - pd.DataFrame()  创建表格
       - df.to_csv()     导出结果

    4. time
       - time.perf_counter()  高精度计时
====================================================================================================
"""
import sys
import os
import time
import importlib.util

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


# ==========================================================================
# 辅助函数：从指定目录加载 predict 函数
# ==========================================================================
def load_predict_fn(model_dir, module_filename):
    """
    从指定模型目录加载 predict 函数，解决不同目录下同名模块的导入冲突。

    为什么不能用简单的 import？
        每个模型目录下都有 config.py，直接 import 会导致后导入的 config
        覆盖先导入的，从而使用错误的配置。

    解决方案：
        使用 importlib.util 从文件路径加载模块，每个模块有独立的命名空间。

    参数：
        model_dir (str):      模型所在目录路径
        module_filename (str): predict_service 文件名

    返回：
        function: predict 函数
    """
    module_path = os.path.join(model_dir, module_filename)
    module_name = os.path.basename(model_dir) + '_predict_service'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    old_path = sys.path.copy()

    # 将模型目录加入 sys.path，让 config.py 等模块可以正确 import
    sys.path.insert(0, model_dir)

    if 'config' in sys.modules:
        del sys.modules['config']
    # 清理可能缓存的同名模块，确保每个模型加载自己目录下的模块
    for key in list(sys.modules.keys()):
        if key.startswith(module_name):
            del sys.modules[key]

    try:
        spec.loader.exec_module(module)
    finally:
        sys.path = old_path

    return module.predict


# ==========================================================================
# 主程序
# ==========================================================================
if __name__ == '__main__':
    # ======================================================================
    # 步骤1：读取测试集和类别映射
    # ======================================================================
    print("=" * 70)
    print("【步骤1】读取测试集数据")
    print("=" * 70)

    # 当前脚本在 06_对比总结/ 下，数据在 ../00-data/data/
    # 将 cwd 切换到项目根目录，确保所有模块的 config.py 中相对路径正确解析
    script_dir = os.path.dirname(__file__) or "."
    project_root = os.path.dirname(script_dir) or "."
    os.chdir(project_root)
    test_path = os.path.join(project_root, "00-data", "data", "test.txt")
    class_path = os.path.join(project_root, "00-data", "data", "class.txt")

    with open(class_path, 'r', encoding='utf-8') as f:
        id_to_class = {i: line.strip() for i, line in enumerate(f)}
        class_to_id = {v: k for k, v in id_to_class.items()}
    print(f"类别数量：{len(id_to_class)}")
    print(f"类别映射：{id_to_class}")

    titles = []
    true_labels_id = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                titles.append(parts[0])
                true_labels_id.append(int(parts[1]))

    print(f"测试集样本数：{len(titles)}")
    print("各类别分布：")
    for lid in sorted(id_to_class.keys()):
        count = true_labels_id.count(lid)
        print(f"  {id_to_class[lid]}: {count}")

    # ======================================================================
    # 步骤2：定义需要对比的模型
    # ======================================================================
    print("\n" + "=" * 70)
    print("【步骤2】加载模型预测服务")
    print("=" * 70)

    models = [
        {
            'name': '随机森林 (RF)',
            'dir': os.path.join(project_root, '01_rf'),
            'file': 'rf_predict_service.py',
            'type': 'predict_service'
        },
        {
            'name': 'XGBoost',
            'dir': os.path.join(project_root, '02_xgb'),
            'file': 'xgb_predict_service.py',
            'type': 'predict_service'
        },
        {
            'name': 'FastText',
            'dir': os.path.join(project_root, '03_fasttext'),
            'file': 'ft_predict_service.py',
            'type': 'predict_service'
        },
        {
            'name': 'BERT',
            'dir': os.path.join(project_root, '04_bert'),
            'file': 'bert_predict_service.py',
            'type': 'predict_service'
        },
        {
            'name': 'BERT-DQ (量化)',
            'dir': os.path.join(project_root, '04_bert'),
            'file': 'bert_predict_dq_service.py',
            'type': 'predict_service',
            'optional': True
        },
        {
            'name': 'BERT-Prune (剪枝)',
            'dir': os.path.join(project_root, '05_模型蒸馏'),
            'file': 'predict_service.py',
            'model_type': 'prune',
            'type': 'distill_service'
        },
        {
            'name': 'BiLSTM (蒸馏)',
            'dir': os.path.join(project_root, '05_模型蒸馏'),
            'file': 'predict_service.py',
            'model_type': 'bilstm',
            'type': 'distill_service'
        },
        {
            'name': 'Transformer (蒸馏)',
            'dir': os.path.join(project_root, '05_模型蒸馏'),
            'file': 'predict_service.py',
            'model_type': 'transformer',
            'type': 'distill_service'
        },
    ]

    all_results = {}
    all_times = {}

    for model_info in models:
        name = model_info['name']
        model_type = model_info.get('type', 'predict_service')
        is_optional = model_info.get('optional', False)

        print(f"\n正在加载 {name}...")

        try:
            if model_type == 'distill_service':
                # 蒸馏服务：使用 predict_service.py 的 predict(text, model_type) 接口
                predict_fn = load_predict_fn(model_info['dir'], model_info['file'])
                dist_model_type = model_info['model_type']
                print(f"  ✓ {name} 加载成功（蒸馏服务）")

                # 蒸馏服务的 get_model() 使用延迟导入（如 from teacher_bert_model import ...），
                # 需要模型目录在 sys.path 中才能找到这些模块
                distill_dir = model_info['dir']
                if distill_dir not in sys.path:
                    sys.path.insert(0, distill_dir)

                test_result = predict_fn(titles[0], dist_model_type)
                print(f"  示例预测：{titles[0][:30]}... → {test_result.get('class', 'N/A')}")

                pred_labels = []
                total_time = 0.0

                for title in titles:
                    start = time.perf_counter()
                    result = predict_fn(title, dist_model_type)
                    elapsed = (time.perf_counter() - start) * 1000
                    pred_labels.append(result.get('class', '未知'))
                    total_time += elapsed

                all_results[name] = pred_labels
                avg_time = total_time / len(titles)
                all_times[name] = avg_time
                print(f"  预测完成，平均耗时：{avg_time:.2f} ms/条")

            else:
                # 传统 predict_service：predict({"title": text}) 接口
                predict_fn = load_predict_fn(model_info['dir'], model_info['file'])
                print(f"  ✓ {name} 加载成功")

                test_result = predict_fn({"title": titles[0]})
                print(f"  示例预测：{titles[0][:30]}... → {test_result.get('pred_class', 'N/A')}")

                pred_labels = []
                total_time = 0.0

                for title in titles:
                    news_data = {"title": title}
                    start = time.perf_counter()
                    result = predict_fn(news_data)
                    elapsed = (time.perf_counter() - start) * 1000
                    pred_labels.append(result.get('pred_class', '未知'))
                    total_time += elapsed

                all_results[name] = pred_labels
                avg_time = total_time / len(titles)
                all_times[name] = avg_time
                print(f"  预测完成，平均耗时：{avg_time:.2f} ms/条")

        except Exception as e:
            if is_optional:
                print(f"  ⚠ {name} 跳过（可选模型不可用）：{e}")
            else:
                print(f"  ✗ {name} 加载失败：{e}")
            all_results[name] = None
            all_times[name] = None

    # ======================================================================
    # 步骤3：计算各模型指标
    # ======================================================================
    print("\n" + "=" * 70)
    print("【步骤3】计算各模型指标")
    print("=" * 70)

    results_table = []

    for model_info in models:
        name = model_info['name']
        pred_labels_name = all_results.get(name)

        if pred_labels_name is None:
            continue

        pred_labels_id = []
        unmapped_count = 0
        for label in pred_labels_name:
            if label in class_to_id:
                pred_labels_id.append(class_to_id[label])
            else:
                pred_labels_id.append(-1)
                unmapped_count += 1

        if unmapped_count > 0:
            print(f"  ⚠ {name}：{unmapped_count} 条无法映射，按错误预测计入")

        # 保持所有模型在完全相同的测试样本上评估，避免指标被抬高。
        f1 = f1_score(
            true_labels_id,
            pred_labels_id,
            labels=list(id_to_class.keys()),
            average='macro',
            zero_division=0,
        )
        acc = accuracy_score(true_labels_id, pred_labels_id)
        prec = precision_score(
            true_labels_id,
            pred_labels_id,
            labels=list(id_to_class.keys()),
            average='macro',
            zero_division=0,
        )
        rec = recall_score(
            true_labels_id,
            pred_labels_id,
            labels=list(id_to_class.keys()),
            average='macro',
            zero_division=0,
        )

        avg_time = all_times.get(name, 0)

        results_table.append({
            '模型': name,
            'F1值': f'{f1:.4f}',
            '准确率': f'{acc:.4f}',
            '精确率': f'{prec:.4f}',
            '召回率': f'{rec:.4f}',
            '推理速度': f'{avg_time:.2f} ms/条'
        })

        print(f"  {name}: F1={f1:.4f}, Acc={acc:.4f}")

    # ======================================================================
    # 步骤4：输出对比表格
    # ======================================================================
    print("\n" + "=" * 70)
    print("【结果汇总】测试集评估对比")
    print("=" * 70)

    if results_table:
        df_results = pd.DataFrame(results_table)
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.width', 120)
        pd.set_option('display.colheader_justify', 'center')
        print(df_results.to_string(index=False))

        # 保存结果
        output_path = os.path.join(script_dir, '模型对比结果.csv')
        df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n对比结果已保存到：{output_path}")

        # ==================================================================
        # 步骤5：分析结论
        # ==================================================================
        print("\n" + "=" * 70)
        print("【分析结论】")
        print("=" * 70)

        best_f1_idx = max(
            range(len(results_table)),
            key=lambda i: float(results_table[i]['F1值'])
        )
        best_f1_model = results_table[best_f1_idx]['模型']
        best_f1_value = results_table[best_f1_idx]['F1值']

        valid_times = [(i, all_times.get(models[i]['name'], float('inf')))
                       for i in range(len(models))
                       if all_times.get(models[i]['name']) is not None]
        if valid_times:
            fastest_idx = min(valid_times, key=lambda x: x[1])[0]
            fastest_model = models[fastest_idx]['name']
            fastest_time = all_times[fastest_model]

        print(f"  🏆 精度最高：{best_f1_model}（F1 = {best_f1_value}）")
        if valid_times:
            print(f"  ⚡ 速度最快：{fastest_model}（{fastest_time:.2f} ms/条）")
        print(f"\n  结论：")
        print(f"  - 传统 ML 方法（RF/XGBoost）速度快，适合实时性要求高的场景")
        print(f"  - 深度学习（BERT）精度最高，但推理速度慢，适合离线批量处理")
        print(f"  - FastText 在精度和速度之间取得了较好的平衡")
        print(f"  - 知识蒸馏（BiLSTM/Transformer）大幅压缩模型的同时保持较高精度")
        print(f"  - 蒸馏模型参数量仅为 BERT 的 1%~15%，推理速度提升 5-20 倍")
        print(f"  - 剪枝 + 蒸馏组合：先剪枝 BERT 做教师，再蒸馏到轻量学生模型")
        print(f"  - 实际部署时需根据业务需求权衡精度、速度与模型体积")

    else:
        print("没有可用的对比结果。请确保所有模型目录下存在对应的 predict_service 文件。")

    print("\n模型对比完成！")