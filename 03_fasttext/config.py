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

        # 2- 预处理后的文件路径：字符级（当前模块下的 data/）
        self.process_char_train_datapath = os.path.join(_module_dir, "data", "process_char_train.txt")
        self.process_char_test_datapath = os.path.join(_module_dir, "data", "process_char_test.txt")
        self.process_char_dev_datapath = os.path.join(_module_dir, "data", "process_char_dev.txt")

        # 3- 预处理后的文件路径：词级（当前模块下的 data/）
        self.process_word_train_datapath = os.path.join(_module_dir, "data", "process_word_train.txt")
        self.process_word_test_datapath = os.path.join(_module_dir, "data", "process_word_test.txt")
        self.process_word_dev_datapath = os.path.join(_module_dir, "data", "process_word_dev.txt")

        # 4- 将 class.txt 文件内容转成字典
        #    格式：{0: 'finance', 1: 'realty', ...}
        #    作用：模型预测出类别ID（数字）后，通过此字典还原为可读的类别名
        #
        # 执行逻辑拆解：
        #   1. open(...)：打开文件，返回【文件迭代器】（每次迭代返回一行）
        #      - 文件迭代器是可迭代对象（Iterable），实现了 __iter__ 和 __next__ 方法
        #      - 每次 for 循环/推导式遍历，自动调用 readline() 读一行，不会一次性加载全部内容
        #   2. enumerate(文件迭代器)：给每一行加序号（从0开始），返回 (序号, 行内容) 元组
        #      - 第1次迭代：(0, 'finance\n')
        #      - 第2次迭代：(1, 'realty\n')
        #   3. 字典推导式 {key: value for ...}：构建字典
        #      - key: enumerate 返回的序号（index），作为类别ID
        #      - value: 行内容的 strip() 结果（去掉换行符），作为类别名
        #
        # 关键细节：
        #   - 必须用 with 包裹：open() 返回的文件对象是可迭代对象，但不会自动关闭
        #     不用 with 会导致【文件句柄泄漏】（占满系统句柄数后会报错）
        #   - strip() 的作用：去掉每行末尾的换行符 \n，否则字典 value 会带换行
        #   - 内存友好：文件迭代器是惰性加载的，即使文件很大也不会 OOM
        with open(self.class_datapath, mode="r", encoding="UTF-8") as f:
            # f 是文件迭代器，被 enumerate 包裹后，被字典推导式遍历
            self.id2label = {index: class_name.strip()
                             for index, class_name in enumerate(f)}

        # 5- 训练好模型的保存路径（当前模块下的 save_model/）
        self.model_char_manual_train = os.path.join(_module_dir, "save_model", "char_manual_train.bin")
        self.model_char_auto_train = os.path.join(_module_dir, "save_model", "char_auto_train.bin")
        self.model_word_manual_train = os.path.join(_module_dir, "save_model", "word_manual_train.bin")
        self.model_word_auto_train = os.path.join(_module_dir, "save_model", "word_auto_train.bin")
        self.legacy_model_word_auto_train = os.path.join(_module_dir, "save_model", "word_auto_train.pkl")

    def resolve_word_auto_model(self):
        """Return the canonical model path, or a compatible historical path."""
        for model_path in (self.model_word_auto_train, self.legacy_model_word_auto_train):
            if os.path.isfile(model_path):
                return model_path
        raise FileNotFoundError(
            "FastText 词级自动调参模型不存在。请先运行 ft_train.py，"
            f"期望路径：{self.model_word_auto_train}"
        )

if __name__ == '__main__':
    config = Config()
    print(config.id2label)