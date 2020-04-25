"""配置管理"""


class CnnConfig(object):
    """CNN 模型超参数和相关配置项"""
    # embedding 层参数
    vocab_size = 10000
    max_len = 1024
    word_dim = 150
    n_class = 2

    # 模型结构
    kernel_sizes=[2,3,4]  # 每层卷积核尺寸
    channel_num = 64      # 每层卷积核数量（通道数）
    drop_rate = 0.5        # Dropout 保留率

    # 优化相关
    LR = 0.01          # 学习率
    STEP_SIZE = 5      # 学习率衰减：步长(每？个epoch衰减)
    GAMMA = 0.1        # 学习率衰减：衰减率
    L2_Reg = 0.01      # L2正则化参数
    CLIP = 6           # 梯度裁剪


class TrainConfig(object):
    """"""
    EPOCH = 100
    train_batch_size = 64
    test_batch_size = 64
    log_interval = 1


class FileConfig(object):
    """"""
    w2v_modelname = "datas/w2v_test.model"
    model_filename = 'model/textCNN_test.pth'
    word_index_filename = 'datas/word_index_test.csv'
    w2v_filename = "datas/w2v_matrix_test.csv"
    train_data = "datas/train_index_test.csv"
    labels_filename = "datas/labels_test.csv"


if __name__=="__main__":
    config=CnnConfig()
    print(config.word_dim)

