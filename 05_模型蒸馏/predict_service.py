"""
====================================================================================================
predict_service.py —— 知识蒸馏模型预测服务封装
====================================================================================================
【文件作用】
    封装所有模型的预测逻辑，供 FastAPI 和 Streamlit 调用。
    统一接口，方便切换模型类型。
====================================================================================================
"""
import torch
from config import Config
from transformers import BertTokenizer

config = Config()
tokenizer = BertTokenizer.from_pretrained(config.bert_path)

# 全局模型缓存，避免重复加载
_model_cache = {}


def get_model(model_type='bilstm'):
    """获取模型（带缓存）"""
    if model_type in _model_cache:
        return _model_cache[model_type]

    if model_type == 'teacher':
        from teacher_bert_model import TeacherBertModel
        model = TeacherBertModel()
        model_path = config.require_model(config.teacher_save_model, 'Teacher 模型')
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))

    elif model_type == 'prune':
        from teacher_bert_model import TeacherBertModel
        model = TeacherBertModel()
        model_path = config.require_model(config.after_prune_path, '剪枝模型')
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))

    elif model_type == 'bilstm':
        from student_bilstm_model import BiLSTMStuModel
        model = BiLSTMStuModel()
        model_path = config.resolve_student_bilstm_model()
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))

    elif model_type == 'transformer':
        from student_transformer_model import TransformerStuModel
        model = TransformerStuModel(return_hidden=False)
        model_path = config.require_model(config.student_transformer_save_model, 'Transformer 学生模型')
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    model = model.to(device=config.device)
    model.eval()
    _model_cache[model_type] = model
    return model


def predict(text, model_type='bilstm'):
    """
    对单条文本进行预测。

    参数:
        text (str):       待预测文本
        model_type (str): 模型类型

    返回:
        dict: {
            'class': str,
            'class_id': int,
            'confidence': float,
            'probabilities': list[dict]
        }
    """
    model = get_model(model_type)

    encoding = tokenizer(
        text,
        max_length=config.max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device=config.device)
    attention_mask = encoding['attention_mask'].to(device=config.device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        logits = output[0] if isinstance(output, tuple) else output
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_class_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[pred_class_id].item()

    probabilities = []
    for i in range(config.class_cnt):
        probabilities.append({
            'class': config.class_list[i],
            'probability': round(probs[i].item(), 4)
        })
    probabilities.sort(key=lambda x: x['probability'], reverse=True)

    return {
        'class': config.class_list[pred_class_id],
        'class_id': pred_class_id,
        'confidence': round(confidence, 4),
        'probabilities': probabilities
    }


def predict_unified(text: str, model_type: str = 'bilstm') -> dict:
    """
    统一预测接口：输入文本字符串，返回统一格式。

    参数:
        text : str — 新闻标题文本
        model_type : str — 模型类型（teacher/prune/bilstm/transformer）

    返回:
        dict — {"label_id": int, "label": str, "confidence": float}
    """
    result = predict(text, model_type)
    return {
        "label_id": result['class_id'],
        "label": result['class'],
        "confidence": result['confidence']
    }


def get_model_info(model_type='bilstm'):
    """获取模型信息"""
    model = get_model(model_type)
    total_params = sum(p.numel() for p in model.parameters())

    if model_type == 'teacher':
        model_name = 'BERT 教师模型'
        model_path = config.teacher_save_model
    elif model_type == 'prune':
        model_name = '剪枝后 BERT 模型'
        model_path = config.after_prune_path
    elif model_type == 'bilstm':
        model_name = 'BiLSTM 学生模型'
        model_path = config.student_bilstm_save_model
    elif model_type == 'transformer':
        model_name = 'Transformer 学生模型'
        model_path = config.student_transformer_save_model

    return {
        'type': model_type,
        'name': model_name,
        'path': model_path,
        'params': total_params,
        'class_cnt': config.class_cnt,
        'class_list': config.class_list,
        'device': config.device
    }