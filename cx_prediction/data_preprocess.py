import pandas as pd
import numpy as np
import arrow
import re
from collections import defaultdict
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from dataObj import DataObj
import jieba
jieba.load_userdict('file/myDict.txt')
import jieba.analyse
import jieba.posseg as pseg

rawDatas_filename="datas/raw_datas_0430.csv"
features_filename="datas/features_0516.csv"
targets_filename="datas/targets_0516.csv"
w2v_filename=""


def jieba_process(text):

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

    # 基于tf-idf 抽取关键词，按照权重大小倒序排列(自动排除权重过低的词)
    # keywords = jieba.analyse.extract_tags(segment, topK=len(wordList), withWeight=True, allowPOS=())
    # new_text = ""
    # for item in keywords:
    #     new_text += item[0] + " "

    return segment


def segment(rawDatas_f):
    startTime = arrow.now()
    df = pd.read_csv(rawDatas_f)
    # 对每一行（每篇文章进行分词操作后,添加到新列‘segment_text’中）
    df['segment_text'] = df['text'].apply(jieba_process)

    # 除去无用样本
    df = df[~df['segment_text'].isin(['无用'])]

    # 保存csv，覆盖原文件
    df.to_csv(rawDatas_f, index=False)
    print("Texts segment finished! Using time:{}".format(arrow.now() - startTime))


def train_word2vec(wordsList):
    startTime=arrow.now()

    # 训练word2vec模型
    model = word2vec.Word2Vec(wordsList, sg=0,  # 默认：CBOW
                              size=150,   # 词向量维度
                              min_count=15,  # 需要计算词向量的最小词频
                              max_vocab_size=None,
                              window=8,   # 窗口大小
                              iter=5,   # SGD 迭代次数（默认）
                              negative=5)  #下采样大小（默认）

    model.save(w2v_filename)

    print("Word2vec model training completed.Using time:{}".format(arrow.now()-startTime))


def document_vectorization(f_rawDatas,f_features,f_targets,tag=True):
    df = pd.read_csv(f_rawDatas)
    df.dropna(inplace=True)
    # 单独保存标签列
    df1 = pd.DataFrame(df['tag'])
    df1.to_csv(f_targets, index=False, header=False)

    """训练 word2vec模型"""
    wordsList = [text.split(" ") for text in df['segment_text']]  # 用于训练word2vec模型，2d list，每篇文章是一个词列表
    # train_word2vec(wordsList)
    # model = word2vec.Word2Vec.load(w2v_filename) # 从.model文件加载
    model=KeyedVectors.load_word2vec_format(w2v_filename) # 从词向量文件加载word2vec
    # 构造一个字典对象  （词：该词的词向量）
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    w2v_words = list(w2v.keys())  # w2v词库中的所有词

    """拟合TfidfVectorizer，得到原始文本各词的idf，生成词权重文件"""
    textList=df['segment_text'].values # 1d list，每篇文章是一个长字符串
    # 先用（分词后的）原始文本拟合TfidfVectorizer，得到每个词的idf值
    tfidf = TfidfVectorizer(analyzer='word')
    tfidf.fit(textList)
    # 设置词权重
    max_idf = max(tfidf.idf_)
    word2weight = defaultdict(lambda: max_idf, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    # 保存词权重到本地文件
    df = pd.DataFrame({"weight": word2weight})
    df.to_csv('datas/word_weight_0516.csv')

    """文档向量转换"""
    startTime = arrow.now()
    # 从词向量构造文档向量
    textList = []
    for words in wordsList:
        text = []
        for w in words:
            # 每个词的词向量取加权平均，生成文档向量
            if w in w2v and w in word2weight:
                text.append(w2v[w] * word2weight[w])
        try:
            text_vector=list(np.mean(text, axis=0))
        except Exception as e:
            # 如果某篇文本中不包括任何 w2v词库中的词，将其文档向量设置成词库中任意随机词的词向量
            ind = np.random.choice(len(w2v_words), 1)[0]
            r_word = w2v_words[ind]
            text_vector=list(w2v[r_word])
        textList.append(text_vector)

    features = pd.DataFrame(textList)
    features.to_csv(f_features,header=False,index=False)

    print("Document vectorization completed! Using time{}".format(arrow.now()-startTime))


if __name__ == "__main__":
    start_time = arrow.now()

    url = "192.168.20.149"
    username = "root"
    password = "admin123!@#"
    db = "text_classification_samples"
    dataObj = DataObj(url, username, password, db)

    # 设定取样数量
    # pos_samples, ratio = dataObj.get_positive_num(), 1
    # sql = "(SELECT title,content,information_type FROM samples_for_analysis where information_type=1) union all " \
    #       "(SELECT title,content,information_type FROM samples_for_analysis where information_type=0 order by rand() limit " \
    # + str(pos_samples * ratio) + ")"
    # print("Sample numbers:{}".format(pos_samples * (1 + ratio)))

    sql = "(SELECT title,content,information_type FROM samples_for_analysis where information_type=1 limit 500)  union all " \
          "(SELECT title,content,information_type FROM samples_for_analysis where information_type=0  limit 500)"
    #print("Sample number:1000")


    """读取数据，存csv"""
    #dataObj.save_csv(sql, rawDatas_filename)

    """分词"""
    #segment(rawDatas_filename)

    """训练word2vec，文档向量转换"""
    document_vectorization(rawDatas_filename,features_filename,targets_filename)

    print("Program completed.Using time:{}".format(arrow.now()-start_time))

