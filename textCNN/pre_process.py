import pandas as pd
import numpy as np
import arrow
from collections import defaultdict
from gensim.models import word2vec
from dataObj import DataObj
from data_preprocess_v3 import jieba_process,train_word2vec

import warnings
warnings.filterwarnings('ignore')


URL = ""
UNAME = ""
PASS = ""
DB = ""


def data_read():
    dataObj = DataObj(URL, UNAME, PASS, DB)
    # 设定取样数量
    # pos_num, ratio = dataObj.get_positive_num(), 1
    pos_num, ratio = 100, 1
    sql = dataObj.set_sql(pos_num, ratio,False)

    data_df=dataObj.read_to_csv(sql, '', save=False)

    # ------- 分词
    data_df['segment_text'] = data_df['text'].apply(jieba_process)
    segment_data = data_df[~data_df['segment_text'].isin(['无用'])]

    segment_data.dropna(inplace=True)
    # 单独保存标签列
    labels= list(segment_data['tag'])

    return segment_data,labels


def train_word2vec(wordsList, w2v_filename):
    startTime=arrow.now()

    # 训练word2vec模型
    model = word2vec.Word2Vec(wordsList, sg=0,  # CBOW sg=0 ; skip-gram sg=1
                              size=150,   # 词向量维度
                              min_count=15,  # 需要计算词向量的最小词频
                              max_vocab_size=None,
                              window=8,   # 窗口大小
                              iter=5,   # SGD 迭代次数（默认）
                              negative=5)  # 下采样大小（默认）

    model.save(w2v_filename)
    return model


def get_weighted_w2v():
    """构造加权词向量"""
    # todo


def get_vocab_vectors(wordsList,w2v_model,word_dim,vocab_size):
    """生成词表，索引，词向量矩阵"""
    # 生成词表
    word_vocb = []
    word_vocb.append('')
    for text in wordsList:
        for word in text:
            word_vocb.append(word)
    word_vocb = set(word_vocb)
    # 用词表的实际大小定义词向量维度（行数）
    # vocab_size=len(word_vocb)

    # 限制词表大小（固定值）
    if len(word_vocb)>vocab_size:
        word_vocb=list(word_vocb)[:vocab_size]

    # 词表与索引的map
    word_to_idx = {word: i for i, word in enumerate(word_vocb)}

    # 生成词向量矩阵（与词表索引对应）
    embedding_matrix = np.zeros((vocab_size, word_dim))
    for word, i in word_to_idx.items():
        if i >= vocab_size:
            continue
        if word in w2v_model.wv:
            embedding_vector = w2v_model.wv[word]
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

    return word_to_idx,embedding_matrix


def get_train_matrix(texts,max_len,word_to_idx):
    """生成以索引表示的全部训练数据，textCNN的输入：每篇文章的词语用其在词表中的索引表示 (2d array)"""
    texts_with_id = np.zeros([len(texts), max_len])
    for i in range(0, len(texts)):
        try:
            if len(texts[i]) < max_len:
                for j in range(0, len(texts[i])):
                    texts_with_id[i][j] = word_to_idx[texts[i][j]]
                for j in range(len(texts[i]), max_len):
                    texts_with_id[i][j] = word_to_idx['']
            else:
                for j in range(0, max_len):
                    texts_with_id[i][j] = word_to_idx[texts[i][j]]
        except Exception as e:
            # 如果文本中的词超出了词表索引范围，直接把索引设为默认值？
            for j in range(0, max_len):
                texts_with_id[i][j]=word_to_idx['']

    texts_with_id = np.array(texts_with_id)


    return texts_with_id


if __name__=="__main__":
    pass






