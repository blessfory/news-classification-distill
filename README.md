# 中文新闻标题分类与轻量化推理

10 类中文新闻标题分类（财经 / 房产 / 股票 / 教育 / 科技 / 社会 / 政治 / 体育 / 游戏 / 娱乐）。
在同一份测试集上对比传统机器学习、预训练模型与压缩模型，并把分类能力封装成 LLM 可调用的工具。

## 模型矩阵

统一评估口径：同一 `test.txt`、同一标签映射、macro-F1（未映射标签按错误计）。

| 模型 | 路线 | F1 | 准确率 | 单条延迟 |
|---|---|---|---|---|
| FastText | 词级 n-gram + softmax | **0.9228** | 0.9228 | **0.026 ms** |
| BERT-Prune | 结构化剪枝（Q/K/V 全局 L1 10%） | 0.8869 | 0.8872 | 8.06 ms |
| BiLSTM（蒸馏） | 知识蒸馏 | 0.8850 | 0.8847 | 5.25 ms |
| Transformer（蒸馏） | 知识蒸馏 + 中间层 MSE | 0.8840 | 0.8840 | 4.36 ms |
| BERT | bert-base-chinese 微调 | 0.8838 | 0.8840 | 4.22 ms |
| BERT-INT8 | 动态量化 | 0.8597 | 0.8604 | 21.46 ms |
| XGBoost | Word2Vec + XGB | 0.7322 | 0.7317 | 1.45 ms |
| 随机森林 | TF-IDF + RF | 0.7275 | 0.7102 | 16.79 ms |

数据见 `06_对比总结/模型对比结果.csv`。FastText 用词级（jieba）预处理，其余模型为字符级，
但都在同一份 `test.txt`、同一标签映射下计算 macro-F1，口径一致。

> FastText 反超 BERT 4 个点，是这个数据集的特点：新闻标题短、关键词信号强，n-gram 特征足够；
> 而 BERT 只训了 5 个 epoch、max_length=32，属于欠拟合状态。真正的差距要看长文本任务。

## 知识蒸馏

- 教师：BERT（**参数冻结，仅分类头可训**），提供软标签与中间层表示
- 学生：BiLSTM / Transformer，从头训练
- 损失：`(1-α)·CE + α·T²·KL`，T=2、α=0.7；Transformer 学生另加 3 层投影后的 hidden-MSE
- 训练：AdamW、cosine 调度、梯度裁剪 1.0、按 dev 选优

**参数量实算**（`sum(p.numel())`）：

| 模型 | 参数量 | 相对 BERT(110M) |
|---|---|---|
| BiLSTM 学生 | 9.62 M（其中 embedding 5.41 M） | 约 1/11 |
| Transformer 学生 | 10.75 M | 约 1/10 |
| BERT 教师 | 约 110 M | 1 |

> 学生模型用 BERT 的词表（21128），embedding 占了 5.4M；扣除后实际「计算部分」约 4.2M。

## 量化实测

INT8 动态量化（`torch.quantization.quantize_dynamic`，仅 Linear 层，qint8）：

- 精度：F1 0.8838 → 0.8597（**-2.4 pt**）
- 延迟：4.22 ms → 21.46 ms

**这个延迟对比不成立**，原因不是量化本身：`quantize_dynamic` 只支持 CPU 推理（fbgemm /
onednn 后端），而原版 BERT 的 4.22 ms 是在 GPU 上测的。两者不是同一基线，不能相减得出
"量化慢 5 倍"。

要评估量化收益，必须在同一设备上跑 CPU-fp32 与 CPU-int8 的对照——这是本项目没做、
需要补的实验。

## 统一预测接口

```python
from unified_predict import predict, predict_batch

predict("NBA季后赛湖人队大胜勇士", model_name='bilstm')
# {'label_id': 7, 'label': 'sports', 'confidence': 0.73}

predict_batch(["央行降准0.5个百分点", "湖人逆转勇士"], model_name='transformer')
```

支持 9 个模型名（rf / xgb / fasttext / bert / bert_dq / bilstm / transformer / teacher / prune）。
各目录下都有同名 `config.py`，接口层用 `importlib` 按路径加载，互不覆盖。

## 运行

```bash
# 1. 预处理 + 训练（以蒸馏为例）
cd 05_模型蒸馏
python data_preprocessing.py
python model_distll.py          # 蒸馏训练
python predict.py               # 单条推理

# 2. 统一对比
cd 06_对比总结
python 模型对比.py

# 3. 服务化
python 04_bert/api_fast_service.py     # FastAPI
streamlit run 04_bert/api_streamlit.py # Web
```

BERT 权重优先读本地 HuggingFace 缓存（`_resolve_bert_path()`），无网络时不会卡在联网检查上。

## 相对基线版本的改动

- **新增 Transformer 学生模型**：基线只有 BiLSTM 学生，补充 Transformer 版本对比蒸馏效果
  （hidden-MSE 对齐 + 3 层投影）。
- **新增 8 模型统一评估**：同一 `test.txt`、同一标签映射、统一 macro-F1，把 FastText 也纳入对比。
- **新增 LLM 兜底链路**（基线完全没有）：低置信度样本调用通义千问二次分类，返回结果
  `confidence=None`（LLM 无概率输出，不伪造置信度）；该能力封装为 LangChain Tool 供 Agent 调用。
- **修正参数量口径**：学生模型实算 9.62M（含 embedding 5.41M），基线注释写的 0.7M 是投影层参数量。
- **修正量化归因**：GPU-fp32 与 CPU-int8 延迟不可直接比较，已在 README 说明并列为待补实验。

## 已知限制

- **FastText 的 Windows 路径坑**：fasttext 的 C++ 层打不开含中文的路径，模型文件放在
  `文本分类项目/03_fasttext/save_model/` 下会报 `cannot be opened for loading`。
  `ft_predict_service.py` 里加了 `_ascii_safe_model()`，先把模型复制到纯英文临时目录再加载。
  训练同理——数据文件也需要先复制到英文路径。
- **教师未微调**：`bert_model.py` 冻结了 BERT 全部参数，只训练分类头，因此教师 F1 0.8838 不是
  BERT 微调的上限，学生"接近教师"的结论要按这个前提理解。
- 量化实验缺少同设备基线：GPU-fp32 与 CPU-int8 的延迟不可比，需补 CPU-fp32 对照。
- 剪枝只做了 Q/K/V 的 10% L1 剪枝，没有 fine-tune，模型体积未变小。
- 5 个模型目录各自复制了一套预处理与 API 服务（69–333 行不等），复用度低。
- 无单元测试；调试信息以 print 为主，未接 logging。
