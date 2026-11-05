"""
====================================================================================================
predict.py —— 知识蒸馏项目推理脚本（支持所有模型）
====================================================================================================
【文件作用】
    加载训练好的模型（教师/学生/剪枝），对单条或批量文本进行类别预测。
    支持交互式命令行输入。

【支持的模型类型】
    --model teacher     : 教师模型（BERT，约 110M 参数）
    --model prune       : 剪枝后教师模型
    --model bilstm      : BiLSTM 学生模型（约 9.6M 参数，默认）
    --model transformer : 3层 Transformer 学生模型（约 15M 参数）

【使用方式】
    python predict.py                        # 默认 BiLSTM
    python predict.py --model teacher        # 教师模型
    python predict.py --model prune          # 剪枝后模型
    python predict.py --model transformer    # Transformer 学生模型
====================================================================================================
"""
import argparse
import torch
from config import Config
from transformers import BertTokenizer

# ==========================================================================
# 全局变量
# ==========================================================================
config = Config()
tokenizer = BertTokenizer.from_pretrained(config.bert_path)


def load_model(model_type='bilstm'):
    """加载指定类型的模型"""
    print(f"加载模型类型: {model_type}")

    if model_type == 'teacher':
        from teacher_bert_model import TeacherBertModel
        model = TeacherBertModel()
        model_path = config.require_model(config.teacher_save_model, 'Teacher 模型')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    elif model_type == 'prune':
        from teacher_bert_model import TeacherBertModel
        model = TeacherBertModel()
        model_path = config.require_model(config.after_prune_path, '剪枝模型')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    elif model_type == 'bilstm':
        from student_bilstm_model import BiLSTMStuModel
        model = BiLSTMStuModel()
        model_path = config.resolve_student_bilstm_model()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    elif model_type == 'transformer':
        from student_transformer_model import TransformerStuModel
        model = TransformerStuModel(return_hidden=False)
        model_path = config.require_model(config.student_transformer_save_model, 'Transformer 学生模型')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    model = model.to(device=config.device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型路径: {model_path}")
    print(f"参数量: {total_params:,}")
    print(f"设备: {config.device}")

    return model


def predict_single(model, text):
    """
    对单条文本进行类别预测。

    返回:
        dict: {
            'class': str,        预测类别名称
            'class_id': int,     预测类别编号
            'confidence': float, 预测置信度
            'probabilities': dict{类别名: 概率}
        }
    """
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

    prob_dict = {
        config.class_list[i]: round(probs[i].item(), 4)
        for i in range(config.class_cnt)
    }

    return {
        'class': config.class_list[pred_class_id],
        'class_id': pred_class_id,
        'confidence': round(confidence, 4),
        'probabilities': prob_dict
    }


def predict_batch(model, texts):
    """批量预测"""
    return [predict_single(model, text) for text in texts]


# ==========================================================================
# 交互式命令行入口
# ==========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="知识蒸馏模型推理")
    parser.add_argument(
        '--model', type=str, default='bilstm',
        choices=['teacher', 'prune', 'bilstm', 'transformer'],
        help='模型类型'
    )
    args = parser.parse_args()

    model = load_model(args.model)

    print("\n" + "=" * 60)
    print(f"【知识蒸馏 - {args.model.upper()} 模型推理】")
    print("=" * 60)
    print(f"类别数量: {config.class_cnt}")
    print(f"类别列表: {config.class_list}")
    print(f"\n输入文本进行预测，输入 'q' 退出\n")

    while True:
        text = input("请输入文本: ").strip()
        if text.lower() == 'q':
            print("退出推理。")
            break
        if not text:
            print("输入为空，请重新输入。\n")
            continue

        result = predict_single(model, text)
        print(f"\n预测结果:")
        print(f"  类别: {result['class']} (ID: {result['class_id']})")
        print(f"  置信度: {result['confidence']:.4f}")

        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        print(f"  Top-3 概率:")
        for cls_name, prob in sorted_probs[:3]:
            bar = '█' * int(prob * 50)
            print(f"    {cls_name:8s}: {prob:.4f} {bar}")
        print()