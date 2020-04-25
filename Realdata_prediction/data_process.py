import pymysql
import pandas as pd
import numpy as np
import arrow
import re
from sklearn.externals import joblib
from dataObj import DataObj
from gensim.models import word2vec
import jieba.posseg as pseg
import jieba
jieba.load_userdict('file/myDict.txt')
import warnings
warnings.filterwarnings("ignore", category=Warning)

dataObj_1 = DataObj("", "", "", "")
dataObj_2 = DataObj("", "", "", "")
weighted_vectors="file/"


def jieba_process(text):
    """
    对单条文本分词
    :param text: 原文本（长字符串）
    :return: 分词后的文本（以空格分隔的长字符串）
    """

    # jieba分词根据词性排除：人名/地名/机构团名/专有名词/时间词/副词/介词/连词/副动词/感叹词/.....
    stop_flags = ['nr', 'ns', 'nt', 'nz', 'p', 'q', 'r', 't', 'tg', 'vd', 'd', 'e', 'f', 'c', 'nrt']

    # 除去特殊字符/英文/数字，仅保留汉字
    text = re.sub(u"([^\u4e00-\u9fa5])", "", text)

    word_cut = pseg.cut(text)
    wordList = []
    for word, flag in word_cut:
        good_word = True
        if len(word) < 2 or word == "\r\n":
            good_word = False
        if flag in stop_flags:
            good_word = False
        if good_word:
            wordList.append(word)
    segment = (" ").join(wordList)  # 分词后重新合并成长字符串
    if len(segment)<2:
        segment="无用"

    return segment


def pre_process(text):
    """
    数据预处理：把单条文本转换为文档向量
#    :param text: 分词后的文本
#    :return: 文档向量（list）
    """
    startTime = arrow.now()

    # 读取加权词向量表
    weighted_w2vs = pd.read_csv(weighted_vectors)

    words = text.split(" ")  # 转化成词列表list
    word_vecs = []
    for word in words:  # 每个词列表中的每个词
        if word in weighted_w2vs.columns:
            word_vecs.append(weighted_w2vs[word])  # 该词的加权词向量放入该篇文章的词向量队列中
    try:
        # 每篇文章的词向量队列求平均，即得到该篇文章的文档向量
        text_feature = list(np.mean(word_vecs, axis=0))
        return text_feature
    except:
        # 如果某文本中没有任何此出现在预训练词库中，无需经过模型预测，直接判为负例
        return "zero"


# def pre_process(text):
#     """
#     数据预处理：把单条文本转换为文档向量
#     :param text: 分词后的文本
#     :return: 文档向量（list）
#     """
#     startTime=arrow.now()
#
#     #读取词权重词典
#     word_weight_1 = pd.read_csv('file/word_weight_0515.csv')
#     word_weight_1.columns = ['word', 'weight']
#
#     # 读取word2vec模型
#     model = word2vec.Word2Vec.load('file/w2v_3.model')
#     # 从word2vec模型构造词向量
#     w2v = dict(zip(model.wv.index2word, model.wv.syn0))
#
#     words = text.split(" ")  # 转化成词列表list
#     word_vecs = []
#     for word in words:  # 每个词列表中的每个词
#         if word in w2v and word in word_weight_1['word'].values:
#             weight = float(word_weight_1[word_weight_1['word'] == word]['weight'].values)
#             word_vecs.append(w2v[word] * weight)  # 该词的加权词向量放入该篇文章的词向量队列中
#     try:
#         # 每篇文章的词向量队列求平均，即得到该篇文章的文档向量
#         text_feature=list(np.mean(word_vecs, axis=0))
#         return text_feature
#     except:
#         # 如果某文本中没有任何此出现在预训练词库中，无需经过模型预测，直接判为负例
#         return "zero"















