import shutil
import tempfile
from pathlib import Path

import jieba

from config import Config
import fasttext
config = Config()


def _ascii_safe_model(path: str) -> str:
    """
    FastText 的 C++ 层在 Windows 下打不开含中文的路径（报
    'cannot be opened for loading'）。这里把模型复制到纯英文的临时目录后再加载。

    只有路径含非 ASCII 字符时才复制，且复制一次后复用缓存。
    """
    src = Path(path)
    if str(src).isascii():
        return str(src)

    cache_dir = Path(tempfile.gettempdir()) / "fasttext_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / src.name

    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)

    return str(dst)


# 1- 加载训练好的模型：因为词级的模型效果最好
model_path = config.resolve_word_auto_model()
try:
    model = fasttext.load_model(_ascii_safe_model(model_path))
except Exception as exc:
    raise RuntimeError(f"无法加载 FastText 模型：{model_path}") from exc

# 2- 预测函数
def predict(news_data):
    # 1- 增加健壮性的代码
    if not news_data.__contains__("title"):
        news_data["error"] = "传递的参数中没有title字段"
        return news_data

    # 2- 取出新闻标题；数据预处理，也就是分词
    title = ' '.join(jieba.lcut(news_data['title']))

    # 3- 预测
    # 返回值类型是嵌套元组。格式：(('__label__science',), array([0.81338769]))
    pred_result = model.predict(title)

    # 4- 抽出结果
    result = pred_result[0][0].replace('__label__','')

    # 5- 返回结果
    news_data['pred_class'] = result

    return news_data


label2id = {v: k for k, v in config.id2label.items()}


def predict_unified(text: str) -> dict:
    """
    统一预测接口：输入文本字符串，返回统一格式。

    参数:
        text : str — 新闻标题文本

    返回:
        dict — {"label_id": int, "label": str, "confidence": float}
    """
    title = ' '.join(jieba.lcut(text))
    pred_result = model.predict(title)
    label_name = pred_result[0][0].replace('__label__', '')
    confidence = float(pred_result[1][0])
    label_id = label2id.get(label_name, -1)

    return {
        "label_id": label_id,
        "label": label_name,
        "confidence": round(confidence, 4)
    }

if __name__ == '__main__':
    news_data = {"title":"体验2D巅峰 倚天屠龙记十大创新概览"}
    # news_data = {"aaaa":"体验2D巅峰 倚天屠龙记十大创新概览"}
    result = predict(news_data)
    print(result)

    result2 = predict_unified("体验2D巅峰 倚天屠龙记十大创新概览")
    print(result2)