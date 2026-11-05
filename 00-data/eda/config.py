# 把常用的变量值放在类中。目的是为了方便使用

class Config:
    def __init__(self):
        # 1- 训练集文件路径
        self.train_datapath = "../data/train.txt"
        # 2- 测试集文件路径
        self.test_datapath = "../data/test.txt"
        # 3- 验证集文件路径
        self.dev_datapath = "../data/dev.txt"

if __name__ == '__main__':
    config = Config()
    print(config.train_datapath)