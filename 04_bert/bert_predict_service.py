"""
====================================================================================================
bert_predict_service.py —— BERT 文本分类预测服务
====================================================================================================
【文件作用】
    加载已训练好的模型，对外提供单条文本分类预测接口。
    输入：{"title": "新闻标题"} 格式的字典
    输出：{"title": "新闻标题", "pred_class": "预测类别名称"} 格式的字典

【依赖包详细说明】
----------------------------------------------------------------------------------------------------
1. torch
   PyTorch 深度学习核心库。
   新增 API：
   - torch.load(path, map_location=None)
       从文件加载用 torch.save() 保存的对象
       参数：
         path:           文件路径（.pkl / .pt / .pth）
         map_location:   指定加载到哪个设备
                         'cpu'：强制加载到 CPU（GPU 训练的模型在 CPU 环境加载时使用）
                         'cuda'：加载到默认 GPU
                         'cuda:0'：加载到指定 GPU
                         None：加载到保存时的原始设备（默认）
       返回：保存时的原始对象（如 state_dict 的 OrderedDict）

   - model.load_state_dict(state_dict, strict=True)
       将参数字典加载到模型中
       参数：
         state_dict: 参数字典（OrderedDict），来自 torch.load() 的返回值
         strict:     是否严格匹配参数名称
                     True：模型和 state_dict 的参数名必须完全一致（默认）
                     False：允许部分不匹配，只加载匹配的参数
       典型用法：
         model = BertClassifierModel()
         model.load_state_dict(torch.load('save_model/bert.pkl'))

   - tensor.item()
       将只包含一个元素的标量张量转为 Python 数值（int/float）
       例：tensor([3]) -> 3

   - torch.no_grad()
       上下文管理器，禁用梯度计算
       预测/推理时必须使用，节省显存和加速

   - torch.argmax(input, dim)
       返回指定维度上最大值的索引
       例：torch.argmax(tensor([[0.1, 0.8, 0.1]]), dim=-1) -> tensor([1])

2. transformers
   - BertTokenizer.from_pretrained(path/name)
       加载预训练 BERT 分词器
       加载后可通过 __call__() 方法对文本进行分词编码

   - BertTokenizer.__call__(texts, max_length, padding, truncation, return_tensors)
       对文本进行分词编码，返回 BatchEncoding 对象
       参数说明：
         texts:          待分词文本，str 或 List[str]
         max_length:     最大序列长度
         padding:        True/'max_length' 填充到 max_length
         truncation:     True 截断超长序列
         return_tensors: 'pt' 返回 PyTorch 张量
       返回 BatchEncoding 对象，包含：
         - input_ids:      token id 序列 (batch, seq_len)
         - attention_mask: 注意力掩码 (batch, seq_len)

3. 本项目其他模块
   - config.Config:              全局配置类（模型路径、类别列表等）
   - bert_model.BertClassifierModel:  BERT 分类模型

【预测流程】
   输入 {"title": "新闻标题"}
     │
     ▼
   取 title 文本
     │
     ▼
   BERT 分词：文本 → token ids + attention mask
     │
     ▼
   模型前向传播：得到各类别 logits
     │
     ▼
   argmax：取 logits 最大的类别索引
     │
     ▼
   索引映射：类别索引 → 类别名称
     │
     ▼
   输出 {"title": "新闻标题", "pred_class": "类别名称"}

【面试高频考点 ⭐】
    1. torch.load 的 map_location 参数作用？（GPU→CPU 跨设备加载）
    2. model.eval() 为什么预测时必须调用？（禁用 Dropout，保证结果稳定）
    3. model.load_state_dict(strict=True/False) 的区别？
    4. torch.argmax 的 dim 参数含义？（沿哪个维度取最大值）
    5. .item() 的作用？（将标量张量转为 Python 数值）
    6. tokenizer 的 return_tensors='pt' 含义？（返回 PyTorch 张量）
    7. 预测时为什么用 torch.no_grad()？（不构建计算图，节省显存，加速推理）
    8. 模型和数据必须在同一设备上，如何保证？
    9. 全局变量加载模型（模块级）的优缺点？（加载一次复用多次 vs 启动慢）
====================================================================================================
"""
from config import Config
from bert_model import BertClassifierModel
from transformers import BertTokenizer
import torch

# ==========================================================================
# 全局变量：加载一次，复用多次，避免每次预测都重新加载模型
# ==========================================================================
config = Config()

# ------------------------------------------------------------------
# 1- 加载已训练好的模型
#
# 步骤：
#   ① 实例化模型（与训练时结构一致）
#   ② model.to(device)：将模型迁移到指定设备（GPU/CPU）
#   ③ torch.load(config.save_model)：从磁盘加载训练时保存的 state_dict
#      - config.save_model 是训练时 torch.save() 保存的文件路径
#   ④ model.load_state_dict(...)：将加载的参数赋值给模型
#
# 注意：
#   - 如果训练时用 GPU 保存，预测时用 CPU 加载，需要加 map_location='cpu'
#     例：torch.load(path, map_location='cpu')
#   - 模型结构必须与训练时完全一致，否则 load_state_dict 会报错
# ------------------------------------------------------------------
model = BertClassifierModel().to(config.device)
model.load_state_dict(torch.load(config.save_model, map_location=config.device))

# ------------------------------------------------------------------
# 2- 加载分词器
# 必须与训练时使用相同的分词器（相同的词表），否则 token id 不一致
# ------------------------------------------------------------------
bert_tokenizer = BertTokenizer.from_pretrained(config.bert_path)

# ------------------------------------------------------------------
# 3- 切换模型为评估模式
#
# model.eval()：
#   必须调用！否则 Dropout 在预测时仍然生效，导致：
#   - 每次预测结果可能不同（随机性）
#   - 预测概率偏低（部分神经元被随机丢弃）
# ------------------------------------------------------------------
model.eval()


# ==========================================================================
# 函数：predict
# 作用：接收单条新闻标题，返回预测的类别名称
# ==========================================================================
def predict(news_data):
    """
    对单条新闻标题进行类别预测。

    参数:
        news_data (dict): 输入字典，格式为 {"title": "新闻标题内容"}

    返回:
        dict: 在原字典基础上增加 "pred_class" 字段
              格式为 {"title": "新闻标题内容", "pred_class": "预测类别名称"}

    示例:
        >>> result = predict({"title": "近期新盘推荐 通州纯新别墅本周开盘"})
        >>> print(result)
        {"title": "近期新盘推荐 通州纯新别墅本周开盘", "pred_class": "房产"}
    """
    # ------------------------------------------------------------------
    # 步骤1：获取新闻标题文本
    #
    # dict.get(key) 和 dict[key] 的区别：
    #   - dict[key]：key 不存在时抛出 KeyError 异常
    #   - dict.get(key)：key 不存在时返回 None（不抛异常）
    #   这里使用 dict[key]，因为调用方必须传入 "title" 字段
    # ------------------------------------------------------------------
    title = news_data["title"]

    # ------------------------------------------------------------------
    # 步骤2：BERT 分词编码
    #
    # bert_tokenizer() 将文本转为 BERT 模型可接受的输入格式：
    #   - 对文本按字切分（中文 BERT 是字级别）
    #   - 添加 [CLS]（开头）和 [SEP]（结尾）特殊标记
    #   - 转换每个 token 为词表中的索引 id
    #   - 填充/截断到 max_length 长度
    #   - 生成 attention_mask（标记有效 token 位置）
    #
    # 参数说明：
    #   max_length=config.max_length: 最大序列长度（来自配置文件）
    #   padding=True:    填充到 max_length（单条文本也会填充到固定长度）
    #   truncation=True: 如果文本超过 max_length，截断到 max_length
    #   return_tensors="pt": 返回 PyTorch 张量格式
    #
    # 注意：这里传入的是单条文本（str），不是列表，但 tokenizer 会自动处理
    #       返回的 input_ids 形状为 (1, max_length)，batch_size=1
    # ------------------------------------------------------------------
    data_tensor = bert_tokenizer(
        title,
        max_length=config.max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # ------------------------------------------------------------------
    # 步骤3：提取分词结果并迁移到指定设备
    #
    # data_tensor.input_ids：
    #   token id 序列，形状 (1, max_length)，即 batch_size=1
    #   例：tensor([[101, 123, 456, ..., 102, 0, 0]])
    #       [CLS]=101, [SEP]=102, [PAD]=0
    #
    # data_tensor.attention_mask：
    #   注意力掩码，形状 (1, max_length)
    #   1=有效 token，0=填充 token（BERT 忽略填充位置）
    #
    # .to(device=config.device)：
    #   将分词结果迁移到与模型相同的设备
    #   模型和数据必须在同一设备上
    # ------------------------------------------------------------------
    input_ids = data_tensor.input_ids.to(device=config.device)
    attention_mask = data_tensor.attention_mask.to(device=config.device)

    # ------------------------------------------------------------------
    # 步骤4：模型预测
    #
    # torch.no_grad()：
    #   禁用梯度计算，预测时必须使用
    #   节省显存，加速前向传播
    # ------------------------------------------------------------------
    with torch.no_grad():
        # 4.1- 前向传播，获取各类别 logits
        # pred_output 形状：torch.Size([1, class_cnt])，即 (batch_size=1, 类别数)
        pred_output = model(input_ids, attention_mask)

        # 4.2- 获取预测类别索引
        #
        # torch.argmax(pred_output, dim=-1)：
        #   沿类别维度（dim=-1 即最后一维）取最大值索引
        #   返回形状：(1,)，即只有一个元素的张量
        #   例：pred_output = [[0.1, 0.8, 0.1]] → argmax → tensor([1])
        #
        # .item()：
        #   将标量张量转为 Python int
        #   tensor([1]) → 1
        pred_index = torch.argmax(pred_output, dim=-1).item()

        # 4.3- 索引映射：类别索引 → 类别名称
        #
        # config.class_list：
        #   训练时从 class.txt 文件加载的类别名称列表
        #   例：['体育', '娱乐', '房产', '科技', ...]
        #   索引 0 对应 '体育'，索引 1 对应 '娱乐'，以此类推
        pred_class = config.class_list[pred_index]

    # ------------------------------------------------------------------
    # 步骤5：组装并返回预测结果
    # 在原字典基础上增加 "pred_class" 字段，方便调用方使用
    # ------------------------------------------------------------------
    news_data["pred_class"] = pred_class
    return news_data


def predict_unified(text: str) -> dict:
    """
    统一预测接口：输入文本字符串，返回统一格式。

    参数:
        text : str — 新闻标题文本

    返回:
        dict — {"label_id": int, "label": str, "confidence": float}
    """
    data_tensor = bert_tokenizer(
        text,
        max_length=config.max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = data_tensor.input_ids.to(device=config.device)
    attention_mask = data_tensor.attention_mask.to(device=config.device)

    with torch.no_grad():
        pred_output = model(input_ids, attention_mask)
        probs = torch.softmax(pred_output, dim=-1).squeeze(0)
        pred_index = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[pred_index].item())

    pred_class = config.class_list[pred_index]

    return {
        "label_id": pred_index,
        "label": pred_class,
        "confidence": round(confidence, 4)
    }


# ==========================================================================
# 测试入口：验证预测服务是否正常工作
# ==========================================================================
if __name__ == '__main__':
    # 测试用例1：模拟一条新闻标题
    test_data1 = {"title": "近期新盘推荐 通州纯新别墅本周开盘"}
    result1 = predict(test_data1)
    print("=" * 60)
    print(f"输入新闻标题: {result1['title']}")
    print(f"预测类别:     {result1['pred_class']}")
    print("=" * 60)

    # 测试用例2：再测试一条
    test_data2 = {"title": "NBA季后赛 湖人队逆转取胜"}
    result2 = predict(test_data2)
    print(f"输入新闻标题: {result2['title']}")
    print(f"预测类别:     {result2['pred_class']}")
    print("=" * 60)

    result3 = predict_unified("近期新盘推荐 通州纯新别墅本周开盘")
    print(f"统一接口测试: {result3}")
    print("=" * 60)