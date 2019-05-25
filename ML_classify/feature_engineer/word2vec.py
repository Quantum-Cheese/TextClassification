"""word2vec 实验：词向量平均化直接生成整篇文章的特征向量"""

import numpy as np
import pandas as pd
import jieba
from gensim.models import word2vec
from gensim.test.utils import common_texts, get_tmpfile


df=pd.DataFrame({'label':[0,0,1,1],'txt':['马斯克造出了载人火箭想要殖民火星','美国队长打不过神奇女侠','坐在这里望着窗外发呆','斯里兰卡人民赋税严重']})

def cutword_1(x):
    words = jieba.cut(x)
    return ' '.join(words)
df['seg_word'] = df.txt.map(cutword_1)


"""训练word2vec词向量模型"""
txt = df.seg_word.values

txtlist = []  # 二维列表
for sent in txt:
    temp = [w for w in sent.split()]
    txtlist.append(temp)

print(txtlist)

num_features = 100
min_word_count = 1
num_workers = 4
context = 20
epoch = 20
sample = 1e-5

# 训练word2vec模型
model = word2vec.Word2Vec(txtlist, workers = num_workers,
                          sample = sample,
                          size = num_features,
                          min_count=min_word_count,
                          window = context,
                          iter = epoch)

model.save("word2vec_vectors")
model = word2vec.Word2Vec.load('word2vec_vectors')

"""利用词向量生成原始文本的特征向量"""

# 1. 直接平均化
def sentvec(sent, m=num_features, model=model):
    res = np.zeros(m)
    words = sent.split() # 输入文本：独立词组成的长字符串（中间用空格分隔）；分割后生成词列表（单篇文章的）
    num = 0

    for w in words:
    # 筛选每个词是否在word2vec训练得出的词汇表index2word里
        if w in model.wv.index2word:
            res += model[w]  # 词向量线性相加
            num += 1.0
    if num == 0:  # 不在词汇表中的词用0代替
        return np.zeros(m)
    else:   # 在词汇表里的词转换成词向量平均值 （词向量线性相加/词汇表里包含的文章中词的个数）
        return res / num


# 2. 使用tfidf进行词频加权计算






n = df.shape[0]
# 构造表示文档的矩阵 行（文章数目），列（word2vec 特征数）
sent_matrix = np.zeros([n, num_features], float)
# 每篇文章依次转换为对应的特征向量（文档矩阵）
for i, sent in enumerate(df.seg_word.values):
    print(i,sent)
    print(type(sent))
    # sent_matrix[i, :] = sentvec(sent)

