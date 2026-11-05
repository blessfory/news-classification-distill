"""
=============================================================================
unified_predict.py —— 统一预测接口（所有模型的单一入口）
=============================================================================
【文件作用】
    将不同模型（RF / XGBoost / FastText / BERT / 蒸馏）的预测调用方式
    统一为一个简单、稳定的 predict(text) 接口，方便后续接入 LLM 兜底、
    Agent、FastAPI 和 Streamlit。

【使用方式】
    from unified_predict import predict
    result = predict("一条新闻标题", model_name='rf')

【支持的模型名称】
    'rf'           — 随机森林（TF-IDF + RF）
    'xgb'          — XGBoost（Word2Vec + XGBoost）
    'fasttext'     — FastText
    'bert'         — BERT
    'bert_dq'      — BERT 动态量化（int8）
    'bilstm'       — BiLSTM 蒸馏模型
    'transformer'  — Transformer 蒸馏模型
    'teacher'      — BERT 教师模型
    'prune'        — BERT 剪枝模型

【统一返回格式】
    {
        "label_id": int,       # 类别编号（0-9）
        "label": str,          # 类别名称（如 'finance'）
        "confidence": float    # 置信度（0.0-1.0）
    }

【设计原则】
    - 不修改现有模型代码的内部逻辑
    - 通过适配层（predict_unified）包装现有预测函数
    - 旧接口保持兼容，不影响现有功能
=============================================================================
"""
import sys
import os
import importlib.util


# ==========================================================================
# 项目根目录（本文件所在目录）
# ==========================================================================
_project_root = os.path.dirname(os.path.abspath(__file__))

# 函数缓存：避免重复加载模块
_model_fn_cache = {}


# ==========================================================================
# 内部工具：从指定目录加载 predict_unified 函数
# ==========================================================================
def _load_predict_unified(model_dir, module_filename):
    """
    使用 importlib.util 从文件路径加载指定模型目录的 predict_unified 函数。

    与 模型对比.py 采用相同策略：避免不同目录下同名模块（config.py）的导入冲突。

    参数：
        model_dir (str):       模型子目录名
        module_filename (str): predict_service 文件名

    返回：
        function: predict_unified 函数
    """
    cache_key = (model_dir, module_filename)
    if cache_key in _model_fn_cache:
        return _model_fn_cache[cache_key]

    module_path = os.path.join(_project_root, model_dir, module_filename)
    module_name = f"unified_{model_dir.replace(os.sep, '_')}_{module_filename.replace('.py', '')}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    old_path = sys.path.copy()
    full_model_dir = os.path.join(_project_root, model_dir)
    sys.path.insert(0, full_model_dir)

    if 'config' in sys.modules:
        del sys.modules['config']
    for key_to_del in list(sys.modules.keys()):
        if key_to_del.startswith(module_name):
            del sys.modules[key_to_del]

    try:
        spec.loader.exec_module(module)
    finally:
        sys.path = old_path

    fn = getattr(module, 'predict_unified', None)
    if fn is None:
        raise AttributeError(
            f"{module_filename} 中未找到 predict_unified 函数，"
            f"请确认已在对应文件中添加该函数"
        )

    _model_fn_cache[cache_key] = fn
    return fn


# ==========================================================================
# 统一预测接口
# ==========================================================================
def predict(text: str, model_name: str = 'rf'):
    """
    统一预测接口：输入新闻标题文本，返回预测结果。

    参数:
        text (str):       新闻标题文本
        model_name (str): 模型名称，可选值：
            'rf'          — 随机森林
            'xgb'         — XGBoost
            'fasttext'    — FastText
            'bert'        — BERT
            'bert_dq'     — BERT 动态量化
            'bilstm'      — BiLSTM 蒸馏模型
            'transformer' — Transformer 蒸馏模型
            'teacher'     — BERT 教师模型
            'prune'       — BERT 剪枝模型

    返回:
        dict — {
            "label_id": int,      类别编号（0-9）
            "label": str,         类别名称
            "confidence": float   置信度
        }

    示例:
        result = predict("同步A股首秀：港股缩量回调", model_name='rf')
        print(result)
        # 输出示例: {'label_id': 0, 'label': 'finance', 'confidence': 0.85}
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text 必须是非空字符串")

    # 传统 ML 模型（各自目录下的 predict_service）
    if model_name == 'rf':
        fn = _load_predict_unified('01_rf', 'rf_predict_service.py')
        return fn(text)

    elif model_name == 'xgb':
        fn = _load_predict_unified('02_xgb', 'xgb_predict_service.py')
        return fn(text)

    elif model_name == 'fasttext':
        fn = _load_predict_unified('03_fasttext', 'ft_predict_service.py')
        return fn(text)

    elif model_name == 'bert':
        fn = _load_predict_unified('04_bert', 'bert_predict_service.py')
        return fn(text)

    elif model_name == 'bert_dq':
        fn = _load_predict_unified('04_bert', 'bert_predict_dq_service.py')
        return fn(text)

    # 蒸馏模型（共享 predict_service，通过 model_type 参数区分）
    elif model_name in ('bilstm', 'transformer', 'teacher', 'prune'):
        # 蒸馏模型的 get_model() 内部使用延迟导入（如 from student_bilstm_model import ...），
        # 需要模型目录在 sys.path 中才能找到这些模块（与 模型对比.py 策略一致）
        distill_dir = os.path.join(_project_root, '05_模型蒸馏')
        if distill_dir not in sys.path:
            sys.path.insert(0, distill_dir)
        fn = _load_predict_unified('05_模型蒸馏', 'predict_service.py')
        return fn(text, model_name)

    else:
        valid_names = [
            'rf', 'xgb', 'fasttext', 'bert', 'bert_dq',
            'bilstm', 'transformer', 'teacher', 'prune'
        ]
        raise ValueError(
            f"不支持的模型名称: '{model_name}'。可选值: {', '.join(valid_names)}"
        )


# ==========================================================================
# 批量预测
# ==========================================================================
def predict_batch(texts: list, model_name: str = 'rf') -> list:
    """
    批量预测。

    参数:
        texts (list):      文本列表
        model_name (str):  模型名称

    返回:
        list[dict]: 预测结果列表
    """
    return [predict(text, model_name) for text in texts]


# ==========================================================================
# 获取可用模型列表
# ==========================================================================
def list_models() -> list:
    """返回所有支持的模型名称列表"""
    return [
        'rf', 'xgb', 'fasttext', 'bert', 'bert_dq',
        'bilstm', 'transformer', 'teacher', 'prune'
    ]


# ==========================================================================
# 带 LLM 兜底的预测
# ==========================================================================
_FALLBACK_THRESHOLD = 0.6


def predict_with_fallback(
    text: str,
    model_name: str = 'rf',
    threshold: float | None = None,
) -> dict:
    """
    传统模型预测 + LLM 低置信度兜底。

    流程:
        1. 传统模型预测，获取 label_id + label + confidence
        2. 如果 confidence >= threshold → 直接返回（source="model"）
        3. 如果 confidence < threshold  → 调用 LLM 兜底（source="llm"）

    参数:
        text (str):              新闻标题文本
        model_name (str):        传统模型名称
        threshold (float|None):  置信度阈值，None 则使用默认值 0.6

    返回:
        dict — {
            "label_id": int,
            "label": str,
            "confidence": float | None,
            "source": "model" | "llm"
        }
    """
    if threshold is None:
        threshold = _FALLBACK_THRESHOLD

    result = predict(text, model_name)
    confidence = result.get("confidence")

    print(f"\n{'=' * 50}")
    print(f"[Predict] 传统模型 ({model_name}) 预测:")
    print(f"  label     = {result['label']}")
    print(f"  label_id  = {result['label_id']}")
    print(f"  confidence = {confidence}")
    print(f"  threshold  = {threshold}")

    # 高置信度：直接返回
    if confidence is not None and confidence >= threshold:
        print(f"\n[Decision] confidence ({confidence}) >= threshold ({threshold}) → 直接返回传统模型结果")
        result["source"] = "model"
        return result

    # 低置信度或无置信度：调用 LLM 兜底
    if confidence is not None:
        print(f"\n[Decision] confidence ({confidence}) < threshold ({threshold}) → 调用 LLM 兜底")
    else:
        print(f"\n[Decision] confidence 为 None → 调用 LLM 兜底")

    from llm_fallback import llm_classify
    llm_result = llm_classify(text)

    print(f"\n[Result] LLM 兜底结果: {llm_result['label']}")
    print(f"{'=' * 50}\n")

    return llm_result


# ==========================================================================
# 测试入口
# ==========================================================================
if __name__ == '__main__':
    test_text = "同步A股首秀：港股缩量回调"

    models_to_test = ['rf', 'xgb', 'fasttext']

    for name in models_to_test:
        try:
            result = predict(test_text, model_name=name)
            print(f"[{name}] {result}")
        except Exception as e:
            print(f"[{name}] 测试失败: {e}")

    print("\n提示：BERT 和蒸馏模型的测试请单独运行（加载较慢）。")