import jieba
import joblib

from config import Config

# 1- 加载配置文件
config = Config()

# 2- 加载训练好的模型
rf = joblib.load(config.rf_model_path)
tfidf = joblib.load(config.tfidf_model_path)

# 3- 预加载类别映射字典（只在启动时读取一次，避免每次预测重复 IO）
#     字典格式：{0: 'finance', 1: 'realty', ...}
with open(config.class_datapath, mode="r", encoding="UTF-8") as f:
    id_class_dict = {i: class_name.strip() for i, class_name in enumerate(f)}
print(f"[INFO] 已加载类别映射：{id_class_dict}")

# 4- 对单条新闻标题进行预测
def predict(news_data):
    """
    对单条新闻标题进行预测
    :param news_data: 字典数据类型。结构：{"title":新闻标题内容}
    :return: 字典。结构：{"title":新闻标题内容,"pred_class":预测的新闻类型}
    """

    # 1- 取出新闻标题；分词；取前30个词；以空格分隔
    words = " ".join(jieba.lcut(news_data["title"])[:30])

    # 2- TF-IDF进行处理
    x = tfidf.transform([words])

    # 3- 模型预测，取第一条（唯一一条）结果
    y_pred_index = rf.predict(x)[0]

    # 4- 通过预加载的 id_class_dict 将预测索引映射为分类名称
    #     字典格式：{0: 'finance', 1: 'realty', ...}
    pred_class_name = id_class_dict[y_pred_index]

    news_data["pred_class"] = pred_class_name

    return news_data


def predict_unified(text: str) -> dict:
    """
    统一预测接口：输入文本字符串，返回统一格式。

    参数:
        text : str — 新闻标题文本

    返回:
        dict — {"label_id": int, "label": str, "confidence": float}
    """
    words = " ".join(jieba.lcut(text)[:30])
    x = tfidf.transform([words])
    y_pred_index = int(rf.predict(x)[0])
    proba = float(rf.predict_proba(x)[0][y_pred_index])
    pred_class_name = id_class_dict[y_pred_index]

    return {
        "label_id": y_pred_index,
        "label": pred_class_name,
        "confidence": round(proba, 4)
    }


if __name__ == '__main__':
    news_data = {"title":"同步A股首秀：港股缩量回调"}
    result = predict(news_data)
    print(result)

    result2 = predict_unified("同步A股首秀：港股缩量回调")
    print(result2)