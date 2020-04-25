"""
完整的训练流程：数据预处理 —— 训练word2v模型和文档向量化（保存加权词向量文件）—- 训练Ensemble模型（报错model文件）
"""
import pandas as pd
import numpy as np
import arrow
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from dataObj import DataObj

from data_preprocess_v3 import jieba_process,train_word2vec
from word2vec_transfer import w2v_transfer
from EnsembleModels import GBDT,RandomForests
import warnings
warnings.filterwarnings('ignore')

URL = "192.168.20.149"
UNAME = "root"
PASS = "admin123!@#"
DB = "text_classification_samples"

features_filename="datas/features_test.csv"
targets_filename="datas/targets_test.csv"
w2v_filename="datas/word2vec/w2v_2test.model"
weighted_w2v="datas/weighted_w2vs_test.csv"


def count_time(startTime,process_name):
    print("{} finished! Using time {}".format(process_name,arrow.now()-startTime))
    return arrow.now()


def get_word_weights(segment_df):
    """拟合TfidfVectorizer，得到原始文本各词的idf，生成词权重文件"""
    textList = segment_df['segment_text'].values  # 1d list，每篇文章是一个长字符串
    # 先用（分词后的）原始文本拟合TfidfVectorizer，得到每个词的idf值
    tfidf = TfidfVectorizer(analyzer='word')
    tfidf.fit(textList)
    # 设置词权重
    max_idf = max(tfidf.idf_)
    word2weight = defaultdict(lambda: max_idf, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    return word2weight


def w2v_transfer(model,weights,save_name):
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    w2v_words = list(w2v.keys())  # w2v词库中的所有词
    # 构造加权词向量（每个词的向量乘以idf权重）
    weighted_w2v = defaultdict(list)
    for word in w2v_words:
        weighted_w2v[word]=weights[word]*w2v[word]

    # 保存加权词向量表
    w_vectors = pd.DataFrame(data=weighted_w2v)
    w_vectors.to_csv(save_name, index=False)


def doc_vecterazation(model,wordsList,word_weights):
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))  # 构造一个字典对象  （词：该词的词向量）
    w2v_words = list(w2v.keys())  # w2v词库中的所有词
    """文档向量转换"""
    # 从词向量构造文档向量
    textList = []
    for words in wordsList:
        text = []
        for w in words:
            # 每个词的词向量取加权平均，生成文档向量
            if w in w2v and w in word_weights:
                text.append(w2v[w] * word_weights[w])
        try:
            text_vector = list(np.mean(text, axis=0))
        except Exception as e:
            # 如果某篇文本中不包括任何 w2v词库中的词，将其文档向量设置成词库中任意随机词的词向量
            ind = np.random.choice(len(w2v_words), 1)[0]
            r_word = w2v_words[ind]
            text_vector = list(w2v[r_word])
        textList.append(text_vector)

    return textList


def data_process(file_targets):
    dataObj = DataObj(URL, UNAME, PASS, DB)
    # 设定取样数量
    # pos_num, ratio = dataObj.get_positive_num(), 1
    pos_num, ratio = 1000, 1
    sql = dataObj.set_sql(pos_num, ratio,False)
    print(sql)

    data_df=dataObj.read_to_csv(sql, '', save=False)

    # ------- 分词
    data_df['segment_text'] = data_df['text'].apply(jieba_process)
    segment_data = data_df[~data_df['segment_text'].isin(['无用'])]

    segment_data.dropna(inplace=True)
    # 单独保存标签列
    df1 = segment_data['tag']
    df1.to_csv(file_targets, index=False, header=False)

    return segment_data


def w2v_vectorization(file_weightedw2v,file_w2v,file_features,file_targets):
    current_time=arrow.now()

    # -- 数据预处理（读取，分词）
    segment_df=data_process(file_targets)
    current_time=count_time(current_time,"datas reading and segment ")

    # 训练 w2v 并保存
    wordsList = [text.split(" ") for text in segment_df['segment_text']]  # 用于训练word2vec模型，2d list，每篇文章是一个词列表
    model = train_word2vec(wordsList, file_w2v)
    current_time = count_time(current_time, "training w2v ")

    # ---- 生成词权重
    word_weights=get_word_weights(segment_df)
    current_time = count_time(current_time, "generating word weights ")

    # ---- 生成加权词向量
    w2v_transfer(model, word_weights, file_weightedw2v)
    current_time = count_time(current_time, "generating weighted word vectors ")

    # ---- 构造文档向量
    textList=doc_vecterazation(model, wordsList, word_weights)
    features = pd.DataFrame(textList)
    features.to_csv(file_features, header=False, index=False)
    _= count_time(current_time, "document vectorization ")


def model_train(model_name,file_features,file_targets):
    if model_name=="GBDT":
        GBDT(file_features,file_targets)
    if model_name=="RF":
        RandomForests(file_features,file_targets)


if __name__ == "__main__":
    startTime=arrow.now()
    w2v_vectorization(weighted_w2v,w2v_filename,features_filename,targets_filename)
    print("Total datas pre processing time {} ".format(arrow.now()-startTime))

    # model_name="RF"
    # startTime = arrow.now()
    # model_train(model_name, features_filename,targets_filename)
    # print("Training {} model finished. Using time {} ".format(model_name,arrow.now()-startTime))

