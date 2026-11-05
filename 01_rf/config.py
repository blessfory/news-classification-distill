# -*- coding: utf-8 -*-
import os


class Config:
    def __init__(self):
        # 路径基准：基于 config.py 自身位置推导绝对路径，不受工作目录影响
        _module_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_module_dir)

        # 1- 原始文件路径（项目根目录下的 00-data/）
        self.train_datapath = os.path.join(_project_root, "00-data", "data", "train.txt")
        self.test_datapath = os.path.join(_project_root, "00-data", "data", "test.txt")
        self.dev_datapath = os.path.join(_project_root, "00-data", "data", "dev.txt")
        self.class_datapath = os.path.join(_project_root, "00-data", "data", "class.txt")
        self.stopwords_datapath = os.path.join(_project_root, "00-data", "data", "stopwords.txt")

        # 2- 预处理后的文件路径（当前模块下的 data/）
        self.process_train_datapath = os.path.join(_module_dir, "data", "process_train_data.csv")
        self.process_test_datapath = os.path.join(_module_dir, "data", "process_test_data.csv")
        self.process_dev_datapath = os.path.join(_module_dir, "data", "process_dev_data.csv")

        # 3- 模型保存路径（当前模块下的 save_model/）
        self.tfidf_model_path = os.path.join(_module_dir, "save_model", "tfidf.pkl")
        self.rf_model_path = os.path.join(_module_dir, "save_model", "rf.pkl")

        # 4- 预测结果路径（当前模块下的 result/）
        self.predict_result_path = os.path.join(_module_dir, "result", "dev.csv")