import joblib
import pandas as pd
from config import Config
from sklearn.metrics import precision_score,recall_score,classification_report,f1_score


# 1- 加载配置文件
config = Config()

if __name__ == '__main__':

    # 2- 加载训练好的模型
    rf = joblib.load(config.rf_model_path)
    tfidf = joblib.load(config.tfidf_model_path)

    # 3- 加载验证集数据
    df = pd.read_csv(config.process_dev_datapath,encoding="UTF-8")
    words = df["words"]
    label = df["label"]

    # 4- 验证集数据特征提取
    x = tfidf.transform(words)
    print(x)

    # 5- 模型预测
    y_pred = rf.predict(x)
    print(y_pred)

    # 6- 评估模型
    print("精确率：", precision_score(label, y_pred, average="macro"))
    print("召回率：", recall_score(label, y_pred, average="macro"))
    print("F1值：", f1_score(label, y_pred, average="macro"))
    print("分类报告：", classification_report(label, y_pred))

    # 7- 预测结果输出到文件
    df["y_pred"] = y_pred
    df.to_csv(config.predict_result_path,mode="w",encoding="UTF-8",index=False,header=True)
