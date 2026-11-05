"""
=============================================================================
config.py —— BERT 文本分类项目配置文件
=============================================================================
【文件作用】集中管理所有超参数、路径、设备等全局配置，方便统一修改和调试。

【依赖包说明】
- torch: PyTorch 深度学习框架，常用功能包括：
    - torch.cuda.is_available()  检查 GPU 是否可用
    - torch.device()             指定运算设备（CPU/GPU）
    - torch.tensor()             创建张量
    - torch.nn.Module            所有神经网络模型的基类
    - torch.optim                 优化器模块（Adam、SGD等）
    - torch.nn.functional         激活函数、损失函数等
    - torch.save() / torch.load() 模型保存与加载

【面试高频考点 ⭐】
    1. torch.cuda.is_available() 的作用？如何判断当前环境是否有 GPU？
    2. batch_size 大小的选择依据是什么？（显存大小、数据集大小、训练稳定性）
    3. max_length 为什么设为 32？BERT 最大支持多少？（512）
    4. 训练集/验证集/测试集的作用和区别？为什么需要三个数据集？
    5. with open() 语句的好处？（自动管理资源，异常安全）
    6. state_dict 是什么？为什么推荐保存 state_dict 而不是整个模型？
    7. 类别不平衡时如何处理？（weighted loss、过采样、欠采样、Focal Loss）
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
    """
    BERT 文本分类全局配置类

    【使用方式】
        config = Config()
        print(config.batch_size)   # 访问属性
        print(config.class_list)   # 类目列表
    """

    def __init__(self):
        # 路径基准：基于 config.py 自身位置推导绝对路径，不受工作目录影响
        _module_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_module_dir)
        # ===================================================================
        # 1- 设备配置
        # torch.cuda.is_available()：检查当前环境是否有可用的 NVIDIA GPU
        #   - 返回 True：有 GPU，使用 'cuda'
        #   - 返回 False：无 GPU，使用 'cpu'
        # torch.device()：创建 device 对象，后续模型和张量可通过 .to(device) 迁移
        # ===================================================================
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

        # ===================================================================
        # 2- 原始数据文件路径
        # train.txt：训练集，每行格式为 "新闻标题\t类别编号"
        # dev.txt：  验证集，用于训练过程中评估模型效果
        # test.txt： 测试集，用于最终评估模型泛化能力
        # class.txt：类别映射文件，每行一个类别名称
        # ===================================================================
        self.train_datapath = os.path.join(_project_root, "00-data", "data", "train.txt")
        self.dev_datapath = os.path.join(_project_root, "00-data", "data", "dev.txt")
        self.test_datapath = os.path.join(_project_root, "00-data", "data", "test.txt")
        self.class_datapath = os.path.join(_project_root, "00-data", "data", "class.txt")

        # ===================================================================
        # 3- 数据加载器（DataLoader）参数
        # batch_size：每个批次加载的样本数量
        #   - 较大值：训练更快，但占用更多显存，梯度估计更稳定
        #   - 较小值：训练较慢，占用显存少，梯度噪声大但有正则化效果
        #   - 常用值：16、32、64、128
        # max_length：BERT 输入的最大序列长度（token 数量）
        #   - BERT-base-chinese 最大支持 512
        #   - 过长会被截断（truncation），过短会被填充（padding）
        #   - 常用值：32、64、128、256、512
        # ===================================================================
        self.batch_size = 64
        self.max_length = 32

        # ===================================================================
        # 4- BERT 预训练模型路径
        # 使用 BertModel.from_pretrained(path) 从本地加载模型
        # 也可以使用在线模型名，如 'bert-base-chinese'，会自动从 HuggingFace 下载
        # ===================================================================
        self.bert_path = _resolve_bert_path()

        # ===================================================================
        # 5- 类别文件解析
        # open(file, mode='r', encoding='utf-8')：
        #   - file: 文件路径
        #   - mode: 'r' 只读, 'w' 写入, 'a' 追加
        #   - encoding: 编码格式，中文推荐 'utf-8'
        # with 语句：自动管理文件关闭，即使发生异常也会关闭文件
        #
        # 列表推导式 [line.strip() for line in f]：
        #   - 遍历文件每一行
        #   - line.strip() 去除每行首尾的空白字符（空格、换行符等）
        #   - 返回处理后的类别名称列表
        # ===================================================================
        with open(self.class_datapath, 'r', encoding='utf-8') as f:
            self.class_list = [line.strip() for line in f]
            self.class_cnt = len(self.class_list)

        # ===================================================================
        # 6- 日志文件存储目录
        # TensorBoard / 训练日志的保存路径
        # ===================================================================
        self.log_path = os.path.join(_module_dir, "logs")

        # ===================================================================
        # 7- 训练好模型的保存路径
        # .pkl 是 PyTorch 常用的模型保存后缀
        # ===================================================================
        self.save_model = os.path.join(_module_dir, "save_model", "bert.pkl")
        self.quantization_model = os.path.join(
            _module_dir, "save_model", "quantization_model.pkl"
        )


if __name__ == '__main__':
    config = Config()
    print("类别列表:", config.class_list)
    print("类别数量:", config.class_cnt)
    print("使用设备:", config.device)