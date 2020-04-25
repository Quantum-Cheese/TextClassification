import torch
import torch.nn as nn
from torch.autograd import Variable


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.vocb_size = args['vocb_size']
        self.word_dim = args['word_dim']
        self.n_class = args['n_class']
        self.max_len = args['max_len']
        self.embedding_matrix = args['embedding_matrix']
        self.kernel_sizes=args['kernel_sizes']
        self.channel_num=args['channel_num']
        self.drop_rate=args['drop_rate']
        
        # 需要将事先训练好的词向量载入
        self.embeding = nn.Embedding(self.vocb_size, self.word_dim, _weight=self.embedding_matrix)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channel_num, kernel_size=self.kernel_sizes[0],
                      stride=1, padding=2),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_num, out_channels=self.channel_num, kernel_size=self.kernel_sizes[1], stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_num, out_channels=self.channel_num, kernel_size=self.kernel_sizes[2], stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dropout = nn.Dropout(self.drop_rate)


    def forward(self, x):
        x = self.embeding(x)
        # print("embeding x ", x.size())
        x = x.view(x.size(0), 1, self.max_len, self.word_dim)
        # print("before conv :", x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # 将（batch，outchanel,w,h）展平为（batch，outchanel*w*h）
        # print("after conv ", x.size())

        # 加入全连接层
        self.out = nn.Linear(x.size(1), self.n_class)
        output = self.out(x)
        return output




