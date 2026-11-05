"""
====================================================================================================
bert_predict_dq_service.py —— BERT 动态量化模型预测服务
====================================================================================================
【文件作用】
    加载 BERT 动态量化模型（int8），对外提供预测接口。
    供 06_对比总结/模型对比.py 调用。

【接口】
    predict(news_data: dict) -> dict
        输入: {"title": "新闻标题"}
        输出: {"pred_class": "类别名", "pred_class_id": 类别编号, "probabilities": [...]}
====================================================================================================
"""
import os
import torch
from config import Config
from transformers import BertTokenizer

config = Config()
tokenizer = BertTokenizer.from_pretrained(config.bert_path)

# 量化模型路径
_QUANTIZATION_PATH = config.quantization_model

# 全局模型缓存
_model = None


def get_model():
    """加载量化模型（带缓存）。

    PyTorch 动态量化的 int8 算子仅在 CPU 上可用，因此量化模型强制使用 CPU。
    """
    global _model
    if _model is not None:
        return _model

    _model = torch.load(_QUANTIZATION_PATH, map_location='cpu', weights_only=False)
    _model.eval()
    return _model


def predict(news_data):
    """
    对单条新闻标题进行预测。

    参数:
        news_data (dict): {"title": "新闻标题"}

    返回:
        dict: {
            "pred_class": "类别名",
            "pred_class_id": 类别编号,
            "probabilities": [{"class": "类别名", "probability": 0.xxx}, ...]
        }
    """
    text = news_data.get('title', '')

    encoding = tokenizer(
        text,
        max_length=config.max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    model = get_model()

    with torch.no_grad():
        logits = model(encoding['input_ids'], encoding['attention_mask'])
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_class_id = torch.argmax(probs, dim=-1).item()

    probabilities = []
    for i in range(config.class_cnt):
        probabilities.append({
            'class': config.class_list[i],
            'probability': round(probs[i].item(), 4)
        })
    probabilities.sort(key=lambda x: x['probability'], reverse=True)

    return {
        'pred_class': config.class_list[pred_class_id],
        'pred_class_id': pred_class_id,
        'probabilities': probabilities
    }


def predict_unified(text: str) -> dict:
    """
    统一预测接口：输入文本字符串，返回统一格式。

    参数:
        text : str — 新闻标题文本

    返回:
        dict — {"label_id": int, "label": str, "confidence": float}
    """
    result = predict({"title": text})
    pred_class_id = result['pred_class_id']
    top_prob = result['probabilities'][0]['probability'] if result['probabilities'] else None

    return {
        "label_id": pred_class_id,
        "label": result['pred_class'],
        "confidence": top_prob
    }