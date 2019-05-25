"""Doc2Vec实验，直接把整篇文章映射成特征向量"""
import pandas as pd
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import jieba


# 构建训练语料库 #
df=pd.read_csv("text_for_test.csv")

#分词
def cutword_1(x):
    words = jieba.cut(x)
    return ' '.join(words)
df['seg_word'] = df.text.map(cutword_1)

# 生成二维列表
text_lists=[text.split(" ") for text in df['seg_word'].values]
#print(text_lists)

#构建gensim指定格式的训练输入 （TaggedDocument对象构成的2d列表）
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(text_lists)]


# 训练文档（句子）向量模型#
model = Doc2Vec(documents, vector_size=10, window=2, min_count=1, workers=4)

model.save('sentenses_vecs.model')

model = Doc2Vec.load('sentenses_vecs.model')
no2=model.docvecs[2] # 取其中一个句子（文章）的向量（numpy array)

print(type(no2))
print(no2.shape)
print(list(no2))



