"""
====================================================================================================
模型量化_DQ.py —— BERT 模型动态量化（Dynamic Quantization）
====================================================================================================
【文件作用】
    对训练好的 BERT 模型进行动态量化（DQ），将模型参数从 float32 降低到 int8，
    以减少模型体积、降低推理延迟，同时尽量保持模型精度。

【什么是模型量化（Quantization）】
    模型量化是将深度学习模型中的浮点数参数（float32）转换为低精度整数（如 int8/qint8）
    的技术。目的是：
    1. 减小模型体积：float32(4字节) → int8(1字节)，模型体积缩小约 4 倍
    2. 加速推理：整数运算比浮点运算更快，尤其在 CPU 上
    3. 降低内存/显存占用：部署到边缘设备（手机、嵌入式）时更友好

    代价：精度会有一定损失（通常 1~3% 以内），需要在精度和速度之间权衡。

【量化类型对比】
    ┌─────────────────────┬──────────────────────┬──────────────────────────────┐
    │ 类型                 │ 特点                  │ 适用场景                      │
    ├─────────────────────┼──────────────────────┼──────────────────────────────┤
    │ 动态量化（DQ）        │ 推理时动态计算量化参数  │ 简单、快速，无需校准数据       │
    │ Dynamic Quantization │ 只量化权重，不量化激活值│ 适合 LSTM、Transformer 等     │
    ├─────────────────────┼──────────────────────┼──────────────────────────────┤
    │ 静态量化（SQ）        │ 需要校准数据预先计算    │ 精度更高，适合 CNN、全连接网络  │
    │ Static Quantization  │ 量化权重和激活值       │ 需要代表性校准数据集           │
    ├─────────────────────┼──────────────────────┼──────────────────────────────┤
    │ 量化感知训练（QAT）    │ 训练时模拟量化效果      │ 精度最高，但需要重新训练        │
    │ QAT                  │ 精度损失最小           │ 对精度要求极高的场景            │
    └─────────────────────┴──────────────────────┴──────────────────────────────┘

【DQ 量化原理】
    1. 只量化模型权重（weights），不量化激活值（activations）
    2. 推理时：权重以 int8 存储，计算时动态反量化回 float32 做矩阵乘法
    3. 量化公式：q = round(w / scale + zero_point)
       - scale：缩放因子，由权重的 min/max 决定
       - zero_point：零点偏移，确保 0 映射到整数 0
    4. 反量化公式：w ≈ (q - zero_point) * scale

【PyTorch 量化引擎】
    - none：   没有硬件加速，纯软件实现
    - onednn：英特尔深度学习加速库（Intel CPU 优化）
    - x86：   x86 架构服务器优化
    - fbgemm：Facebook 的量化计算引擎（x86 CPU 优化，默认推荐）

【面试高频考点 ⭐】
    1. 什么是模型量化？为什么要量化？
    2. 动态量化 vs 静态量化 vs QAT 的区别和适用场景
    3. int8 和 qint8 的区别是什么？
    4. 量化后精度下降的原因？如何缓解？
    5. 哪些层适合量化？哪些不适合？（如 nn.Linear 适合，LayerNorm 不适合）
    6. 量化后模型如何在 CPU 上推理？（GPU 不支持 int8 矩阵乘法）
    7. 量化对模型体积的影响：float32(4B) → int8(1B)，缩小约 4 倍

【⚠️ 注意事项】
    1. DQ 量化必须在 CPU 上运行！GPU 不支持 int8 量化推理
    2. 量化后的模型推理时也必须放在 CPU 上
    3. 量化只对特定层类型有效（如 nn.Linear），其他层保持不变
    4. 量化后模型保存格式与普通模型不同，加载时需注意

【依赖包详细说明】
----------------------------------------------------------------------------------------------------
1. torch
   PyTorch 深度学习核心库。
   量化相关 API：
   - torch.backends.quantized.supported_engines
       查看当前环境支持的量化引擎列表
       返回：['none', 'onednn', 'x86', 'fbgemm']（取决于硬件和编译选项）

   - torch.backends.quantized.engine
       设置/查看当前使用的量化引擎
       可选值：'none', 'onednn', 'x86', 'fbgemm'

   - torch.quantization.quantize_dynamic(model, qconfig_spec, dtype)
       对模型进行动态量化
       参数：
         model:         要量化的模型实例（nn.Module）
         qconfig_spec:  指定量化哪些层类型，类型为 set/dict
                        例：{torch.nn.Linear} 量化所有线性层
                            {torch.nn.Linear, torch.nn.LSTM} 量化线性层和LSTM
                        设为 None 时默认量化所有支持的层
         dtype:         量化后的数据类型
                        torch.qint8：PyTorch 专用 8 位量化整数（推荐）
                        torch.float16：半精度浮点（不是真正的量化，但能减小体积）
       返回：量化后的模型副本（原模型不变）

   - torch.save(obj, path)
       保存量化模型
       注意：量化模型的保存和加载与普通模型类似，但内部结构不同

   - torch.load(path, map_location)
       加载模型
       参数 map_location='cpu'：强制加载到 CPU（量化模型必须在 CPU 上）

2. 本项目其他模块
   - config.Config:              全局配置类
   - bert_model.BertClassifierModel:  BERT 分类模型
   - bert_train_eval.eval_model:  模型评估函数

【量化前后效果对比（参考）】
    量化前：f1score=0.8777, accuracy=0.8777, precision=0.8790, recall=0.8777
    量化后：f1score=0.8399, accuracy=0.8404, precision=0.8437, recall=0.8404
    精度损失：约 3~4%，模型体积缩小约 4 倍
====================================================================================================
"""
from config import Config
from bert_model import BertClassifierModel
from bert_train_eval import eval_model
import torch


# ==========================================================================
# 程序入口
# ==========================================================================
if __name__ == '__main__':

    # ======================================================================
    # 步骤1：【了解】查看支持的量化引擎
    #
    # torch.backends.quantized.supported_engines：
    #   返回当前 PyTorch 编译时启用的量化引擎列表
    #   不同平台/编译选项下返回的引擎可能不同
    #
    # 常见引擎：
    #   none    — 无硬件加速，纯软件实现，通用但最慢
    #   onednn  — 英特尔 oneDNN 深度学习加速库，Intel CPU 优化
    #   x86     — x86 架构服务器优化，使用 FBGEMM 后端
    #   fbgemm  — Facebook 通用矩阵乘法库，x86 CPU 优化
    #
    # ⭐ 面试考点：PyTorch 有哪些量化引擎？各自适用什么场景？
    # ======================================================================
    print("=" * 60)
    print("【步骤1】查看支持的量化引擎")
    print("=" * 60)
    print("支持的量化引擎：", torch.backends.quantized.supported_engines)

    # ======================================================================
    # 步骤2：【可选】手动设置量化引擎
    #
    # torch.backends.quantized.engine：
    #   设置当前使用的量化引擎
    #   如果不手动设置，PyTorch 会自动根据硬件环境选择最优引擎
    #
    # 不同引擎的适用场景：
    #   'fbgemm' — x86 CPU，推荐用于 Intel/AMD 服务器
    #   'qnnpack' — ARM CPU，推荐用于移动端/嵌入式设备
    #   'onednn' — Intel CPU，Intel 深度学习加速
    #
    # ⭐ 面试考点：为什么要手动设置量化引擎？什么时候需要？
    #   答：跨平台部署时，需要确保量化行为一致；
    #       在特定硬件上手动选择最优引擎以获得最佳性能
    # ======================================================================
    print("\n" + "=" * 60)
    print("【步骤2】设置量化引擎")
    print("=" * 60)
    torch.backends.quantized.engine = 'onednn'
    print(f"当前量化引擎：{torch.backends.quantized.engine}")

    # ======================================================================
    # 步骤3：加载配置
    #
    # Config() 加载全局配置，包括：
    #   - device:      运算设备（自动检测 GPU/CPU）
    #   - bert_path:   BERT 预训练模型路径
    #   - save_model:  训练好的模型保存路径
    #   - class_list:  类别名称列表
    #   - class_cnt:   类别数量
    #   - max_length:  最大序列长度
    #   - batch_size:  批次大小
    #
    # ⚠️ 关键：DQ 量化必须在 CPU 上运行，强制设置 device='cpu'
    #   原因：GPU 不支持 int8 量化推理，量化操作只能在 CPU 上执行
    # ======================================================================
    print("\n" + "=" * 60)
    print("【步骤3】加载配置")
    print("=" * 60)
    config = Config()
    # ⚠️ 强制使用 CPU：DQ 量化必须在 CPU 上运行
    config.device = 'cpu'
    print(f"运算设备：{config.device}（已强制设为 CPU，DQ 量化必须用 CPU）")
    print(f"模型路径：{config.save_model}")
    print(f"类别数量：{config.class_cnt}")

    # ======================================================================
    # 步骤4：评估量化前的模型效果
    #
    # 目的：获取量化前的基准指标，量化后对比精度损失
    #
    # 流程：
    #   4.1- 创建模型实例（与训练时结构一致）
    #   4.2- 加载训练好的模型权重（map_location='cpu' 强制加载到 CPU）
    #   4.3- 设置评估模式（禁用 Dropout 等）
    #   4.4- 在验证集上评估，获取 F1/准确率/精确率/召回率
    # ======================================================================
    print("\n" + "=" * 60)
    print("【步骤4】评估量化前的模型效果")
    print("=" * 60)

    # 4.1- 创建模型实例对象
    # BertClassifierModel() 会加载预训练 BERT 并冻结参数
    model = BertClassifierModel()

    # 4.2- 加载训练好的模型权重
    #
    # torch.load() 参数详解：
    #   config.save_model:  模型文件路径（.pkl）
    #   map_location='cpu': 将模型参数强制加载到 CPU
    #                       原因：
    #                       1. GPU 训练的模型，在 CPU 环境加载需要此参数
    #                       2. DQ 量化必须在 CPU 上进行
    #                       可选值：
    #                         'cpu'       — 加载到 CPU
    #                         'cuda'      — 加载到默认 GPU
    #                         'cuda:0'    — 加载到指定 GPU
    #                         None        — 加载到保存时的原始设备（默认）
    #                         torch.device('cpu') — 使用 device 对象
    #
    # model.load_state_dict() 参数详解：
    #   state_dict:  torch.load() 返回的参数字典（OrderedDict）
    #   strict=True: 是否严格匹配参数名（默认 True）
    #                True：模型和 state_dict 参数名必须完全一致
    #                False：允许部分匹配，只加载匹配的参数
    #
    # ⭐ 面试考点：torch.load 的 map_location 参数作用？
    #   答：将模型参数加载到指定设备，解决 GPU→CPU 跨设备加载问题
    # ======================================================================
    model.load_state_dict(torch.load(config.save_model, map_location='cpu'))

    # 4.3- 设置评估模式
    #
    # model.eval() vs model.train()：
    #   eval()：禁用 Dropout（不丢弃神经元）、BatchNorm 使用全局统计量
    #   train()：启用 Dropout、BatchNorm 使用批次统计量
    #   评估/推理必须调用 eval()，否则结果不稳定
    #
    # ⭐ 面试考点：model.eval() 和 model.train() 的区别？
    #   答：影响 Dropout 和 BatchNorm 的行为
    #       eval() 下 Dropout 不生效，BatchNorm 用训练时累积的全局均值/方差
    # ======================================================================
    model.eval()

    # 4.4- 评估量化前的模型
    #
    # eval_model() 返回 4 个指标：
    #   f1score:   F1 值（macro 平均），精确率和召回率的调和平均
    #   accuracy:  准确率，预测正确的比例
    #   precision: 精确率（macro 平均），预测为正中真正为正的比例
    #   recall:    召回率（macro 平均），真实为正中被预测出来的比例
    #
    # ⭐ 面试考点：F1、Precision、Recall 的区别和计算公式？
    #   Precision = TP / (TP + FP)  — 预测了多少真正的正类
    #   Recall = TP / (TP + FN)     — 找到了多少真正的正类
    #   F1 = 2 * P * R / (P + R)    — 精确率和召回率的调和平均
    # ======================================================================
    f1score, accuracy, precision, recall = eval_model(model)
    print(f"\n量化前的指标：")
    print(f"  F1值 = {f1score:.4f}")
    print(f"  准确率 = {accuracy:.4f}")
    print(f"  精确率 = {precision:.4f}")
    print(f"  召回率 = {recall:.4f}")

    # ======================================================================
    # 步骤5：模型量化（DQ 动态量化）
    #
    # torch.quantization.quantize_dynamic() 参数详解：
    #
    #   model (nn.Module)：
    #       要量化的模型实例
    #       注意：传入的是原始模型，函数会返回量化后的副本，原模型不变
    #
    #   qconfig_spec (set/dict/None)：
    #       指定对哪些层类型进行量化
    #       类型：Set[type] 或 Dict[type, QConfig]
    #       例：
    #         {torch.nn.Linear}                    — 量化所有线性层
    #         {torch.nn.Linear, torch.nn.LSTM}     — 量化线性层和 LSTM
    #         {torch.nn.Linear, torch.nn.Conv2d}   — 量化线性层和卷积层
    #         None                                  — 默认量化所有支持的层
    #
    #       为什么只量化 nn.Linear？
    #       - BERT 中大部分计算密集在 nn.Linear（多头注意力中的 Q/K/V/O 投影、
    #         前馈网络等），量化这些层收益最大
    #       - LayerNorm、Embedding 等层不适合量化（精度损失大）
    #       - 激活函数（GELU）不量化
    #
    #       ⭐ 面试考点：为什么只量化 Linear 层，不量化 Embedding/LayerNorm？
    #       答：Embedding 层查表操作对精度敏感；LayerNorm 归一化操作对精度敏感
    #
    #   dtype (torch.dtype)：
    #       量化后的数据类型
    #       torch.qint8：PyTorch 专用 8 位有符号量化整数（推荐）
    #                     范围：[-128, 127]
    #                     特点：专为量化设计，支持加速的量化算子
    #       其他可选值：
    #         torch.float16 — 半精度浮点（不是真正的量化，但也能减小体积）
    #         torch.int8    — 标准 8 位整数（不推荐，PyTorch 量化用 qint8）
    #
    #       ⭐ 面试考点：qint8 和 int8 的区别？
    #       答：qint8 是 PyTorch 专门为量化设计的数据类型，内部有 scale 和
    #           zero_point 等量化参数；int8 是标准整数类型，没有量化语义
    #
    # 量化内部流程：
    #   1. 遍历模型的所有子模块
    #   2. 对匹配 qconfig_spec 中类型的层，进行量化转换
    #   3. 量化层：权重存储为 int8，前向传播时动态反量化回 float32 计算
    #   4. 非量化层：保持原样
    #
    # 量化后模型结构变化：
    #   nn.Linear → nn.quantized.dynamic.Linear（动态量化线性层）
    #   权重从 torch.float32(Tensor) → torch.qint8(QuantizedTensor)
    # ======================================================================
    print("\n" + "=" * 60)
    print("【步骤5】执行 DQ 动态量化")
    print("=" * 60)
    print(f"量化目标层类型：{{torch.nn.Linear}}")
    print(f"量化数据类型：torch.qint8")

    quantization_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={torch.nn.Linear},
        dtype=torch.qint8
    )

    print("量化完成！")

    # ======================================================================
    # 步骤6：评估量化后的模型效果
    #
    # 目的：对比量化前后的精度变化，评估量化带来的精度损失
    #
    # 注意：量化后的模型同样需要 eval() 模式
    # ======================================================================
    print("\n" + "=" * 60)
    print("【步骤6】评估量化后的模型效果")
    print("=" * 60)

    # 设置评估模式
    quantization_model.eval()

    # 评估量化后的模型
    f1score, accuracy, precision, recall = eval_model(quantization_model)
    print(f"\n量化后的指标：")
    print(f"  F1值 = {f1score:.4f}")
    print(f"  准确率 = {accuracy:.4f}")
    print(f"  精确率 = {precision:.4f}")
    print(f"  召回率 = {recall:.4f}")

    # ======================================================================
    # 步骤7：打印量化后的模型结构
    #
    # 对比量化前后的模型结构变化：
    #   量化前：Linear(in_features=768, out_features=10, bias=True)
    #   量化后：DynamicQuantizedLinear(in_features=768, out_features=10, ...)
    #
    # 观察要点：
    #   1. nn.Linear 被替换为 DynamicQuantizedLinear
    #   2. 权重参数类型从 float32 变为 qint8
    #   3. 其他层（BertModel、LayerNorm 等）保持不变
    #
    # ⭐ 面试考点：量化后模型结构发生了什么变化？
    #   答：nn.Linear → nn.quantized.dynamic.Linear
    #       权重从 float32 的 Tensor 变为 qint8 的 QuantizedTensor
    #       前向传播时自动反量化回 float32 做计算
    # ======================================================================
    print("\n" + "=" * 60)
    print("【步骤7】量化后的模型结构")
    print("=" * 60)
    print(quantization_model)

    # ======================================================================
    # 步骤8：保存量化后的模型
    #
    # torch.save() 保存量化模型：
    #   与保存普通模型语法相同，但内部结构不同
    #   量化模型保存时会同时保存量化参数（scale、zero_point）
    #
    # 加载量化模型：
    #   quantized_model = torch.load('quantization_model.pkl')
    #   注意：加载量化模型时不需要先实例化原始模型，
    #         torch.load 会直接加载完整的量化模型
    #
    # ⭐ 面试考点：量化模型如何保存和加载？
    #   答：保存和普通模型一样用 torch.save/load
    #       但加载时不需要先创建模型实例，直接 load 即可
    #       必须在 CPU 上加载和推理
    # ======================================================================
    print("\n" + "=" * 60)
    print("【步骤8】保存量化模型")
    print("=" * 60)
    save_path = config.quantization_model
    torch.save(quantization_model, save_path)
    print(f"量化模型已保存至：{save_path}")

    # ======================================================================
    # 步骤9：对比总结
    # ======================================================================
    print("\n" + "=" * 60)
    print("【总结】量化前后对比")
    print("=" * 60)
    print("量化效果评估：")
    print("  - 模型体积：缩小约 4 倍（float32 → int8）")
    print("  - 推理速度：量化模型只能在 CPU 上推理，原版 BERT 跑在 GPU 上")
    print("    （本项目 4.22ms 为 GPU 版，21.46ms 为 CPU 量化版，两者口径不同，不可直接相减）")
    print("  - 要评估量化收益：需在同一设备上比较 CPU-fp32 与 CPU-int8")
    print("  - 精度损失：本项目 test 集 F1 0.8838 → 0.8597（-2.4pt）")
    print("  - 适用场景：CPU 推理、边缘设备部署、模型压缩（上线前须实测延迟）")
    print("\n⚠️ 注意事项：")
    print("  1. 量化模型必须在 CPU 上运行，不能放 GPU")
    print("  2. 量化后模型体积减小，但推理时仍需 CPU 内存")
    print("  3. 不同量化引擎（fbgemm/onednn）性能有差异，建议测试后选择")
    print("=" * 60)