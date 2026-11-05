"""
=============================================================================
config.py —— BERT 文本分类项目配置文件
=============================================================================
【文件作用】集中管理所有超参数、路径、设备等全局配置。

【路径约定】所有脚本请从项目根目录（文本分类项目/）运行，例如：
           python 05_模型蒸馏/model_distll.py
=============================================================================
"""
import os
import pathlib
import torch


def _resolve_bert_path() -> str:
    """
    解析 BERT 预训练权重路径。

    优先顺序：
      1. 环境变量 BERT_MODEL_PATH（显式指定本地目录）
      2. HuggingFace 本地缓存中的 bert-base-chinese（离线可用，避免联网检查超时）
      3. 退回模型名，由 transformers 在线下载
    """
    env_path = os.environ.get("BERT_MODEL_PATH")
    if env_path:
        return env_path

    hub = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    snapshots = sorted(hub.glob("models--bert-base-chinese/snapshots/*"))
    if snapshots:
        return str(snapshots[-1])

    return "bert-base-chinese"


class Config:

    def __init__(self):
        # ===================================================================
        # 0- 路径基准：基于 config.py 自身位置推导绝对路径，不受工作目录影响
        #    config.py 位于 05_模型蒸馏/ 下
        # ===================================================================
        _module_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_module_dir)

        # ===================================================================
        # 1- 设备配置
        # ===================================================================
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

        # ===================================================================
        # 2- 数据文件路径
        # ===================================================================
        self.train_datapath = os.path.join(_project_root, "00-data", "data", "train.txt")
        self.dev_datapath = os.path.join(_project_root, "00-data", "data", "dev.txt")
        self.test_datapath = os.path.join(_project_root, "00-data", "data", "test.txt")
        self.class_datapath = os.path.join(_project_root, "00-data", "data", "class.txt")

        # ===================================================================
        # 3- 数据加载器参数
        # ===================================================================
        self.batch_size = 64
        self.max_length = 32

        # ===================================================================
        # 4- BERT 预训练模型路径
        # ===================================================================
        self.bert_path = _resolve_bert_path()

        # ===================================================================
        # 5- 类别列表
        # ===================================================================
        with open(self.class_datapath, 'r', encoding='utf-8') as f:
            self.class_list = [line.strip() for line in f]
            self.class_cnt = len(self.class_list)

        # ===================================================================
        # 6- 日志路径
        # ===================================================================
        self.log_path = _module_dir

        # ===================================================================
        # 7- 模型保存路径（当前模块下的 save_model/）
        # ===================================================================
        self.teacher_save_model = os.path.join(_module_dir, "save_model", "bert.pkl")
        self.student_bilstm_save_model = os.path.join(_module_dir, "save_model", "student_bilstm.pkl")
        self.legacy_student_bilstm_save_model = os.path.join(_module_dir, "save_model", "student_bert.pkl")
        self.student_transformer_save_model = os.path.join(_module_dir, "save_model", "student_transformer.pkl")
        self.after_prune_path = os.path.join(_module_dir, "save_model", "prune_bert.pkl")

    def resolve_student_bilstm_model(self):
        """Return the current BiLSTM path, falling back to the historical name."""
        for model_path in (self.student_bilstm_save_model, self.legacy_student_bilstm_save_model):
            if os.path.isfile(model_path):
                return model_path
        raise FileNotFoundError(
            "BiLSTM 学生模型不存在。请先运行 model_distll.py，"
            f"期望路径：{self.student_bilstm_save_model}"
        )

    @staticmethod
    def require_model(model_path, model_name):
        """Raise a clear error before attempting to load a missing checkpoint."""
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"{model_name}不存在：{model_path}")
        return model_path


if __name__ == '__main__':
    config = Config()
    print("类别列表:", config.class_list)
    print("类别数量:", config.class_cnt)
    print("使用设备:", config.device)
    print("训练数据:", config.train_datapath)
    print("验证数据:", config.dev_datapath)
    print("测试数据:", config.test_datapath)