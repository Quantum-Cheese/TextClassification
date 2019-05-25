import pymysql
import pandas as pd
import numpy as np
import arrow
import re
from sklearn.externals import joblib
from dataObj import DataObj
from gensim.models import word2vec
import warnings
warnings.filterwarnings('ignore')
import jieba.posseg as pseg
import jieba
jieba.load_userdict('file/myDict.txt')

dataObj_150 = DataObj("192.168.20.150", "root", "les-admin-cx", "les_crawler_data_analysis")
dataObj_149 = DataObj("192.168.20.149", "root", "admin123!@#", "qh_Realtime_Test")



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
    :param text: 分词后的文本
    :return: 文档向量（list）
    """
    startTime=arrow.now()

    #读取词权重词典
    word_weight_1 = pd.read_csv('file/word_weight_0515.csv')
    word_weight_1.columns = ['word', 'weight']

    # 读取word2vec模型
    model = word2vec.Word2Vec.load('file/w2v_3.model')
    # 从word2vec模型构造词向量
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))

    words = text.split(" ")  # 转化成词列表list
    word_vecs = []
    for word in words:  # 每个词列表中的每个词
        if word in w2v and word in word_weight_1['word'].values:
            weight = float(word_weight_1[word_weight_1['word'] == word]['weight'].values)
            word_vecs.append(w2v[word] * weight)  # 该词的加权词向量放入该篇文章的词向量队列中
    try:
        # 每篇文章的词向量队列求平均，即得到该篇文章的文档向量
        text_feature=list(np.mean(word_vecs, axis=0))
        return text_feature
    except:
        # 如果某文本中没有任何此出现在预训练词库中，无需经过模型预测，直接判为负例
        return "zero"


"""单进程（顺序）预测批量数据"""
def predict(model_name,raw_data):
    """
    加载模型预测一批数据（从参数传过来的 DataFrame），并把结果保存到数据库
    :param model_name:
    :param raw_data:
    :return:
    """

    # 加载模型
    model = joblib.load(model_name)
    conn = pymysql.connect(host="192.168.20.149", user="root", passwd="admin123!@#", db="qh_Realtime_Test",
                           charset='utf8')
    curs = conn.cursor()

    for n in range(raw_data.shape[0]):
        startTime0 = arrow.now()

        try:
            data=raw_data.iloc[n]
        except Exception as e:
            print(e)
            continue
        text_vector=pre_process(data['segment_text'])  # 文档向量
        if text_vector == "zero":  # 如果文本所有词语都未出现在预训练词库中，直接判为0
            prediction = 0
        else:
            prediction = model.predict([text_vector])  # 否则，输入文档向量，用模型预测
        try:
            # 预测结果保存到数据表 ,保留源表中所有字段，在分类字段填入预测类型（0或1）
            sql = "insert into GBDT_Predict_realtime_1 (content_id,id,text,gmt_crawler,Tag) values (%s, %s,'%s','%s',%s)"
            curs.execute(sql % (int(data['contnet_id']), int(data['id']),pymysql.escape_string(data['text']),
                                str(data['gmt_crawler']),int(prediction[0])))
            conn.commit()
            # todo：正式程序，把插入结果表的操作改为直接更新源表的 tag_i 字段
        except Exception as e:
            # 记录下未分析成功的数据id，和错误原因
            print("Insert error:{}.  Data id:{} analysis failed".format(e,data['id']))
            #logging.error("Insert error:{}.  Data id:{} analysis failed".format(e,data['id']))

    conn.close()


"""多进程批量预测"""
def multi_predict(model_name,raw_data):
    pass


def update_lastId(orgData_table,start_id):

    """取批量数据（从上次的last_id到目前的所有新数据）"""
    sql = "select id from " + orgData_table + " where id> " + str(start_id)
    ids = dataObj_150.data_read(sql)

    # 若无新数据，返回原来的start_id (last_id不变）
    if len(ids)<1:
        return start_id

    # 有新数据，返回最后一条的id，用其更新last_id
    return ids[-1][0]


def iter_run(orgData_table,model_name,start_id):
    """
    :param orgData_table:  源数据表名
    :param model_name:     预测用的ML模型（pkl文件名）
    :param start_id:      上次取到的最新数据id，本次分析从last_id+1开始取到当前最新数据
    :return:              分析完后最后一条的id，分析的数据条数
    """

    """取批量数据（从上次的last_id到目前的所有新数据）"""
    sql_2 = "select content_id,id,title,content,gmt_crawler from "+orgData_table+" where id> "+str(start_id)
    raw_datas = dataObj_150.data_read(sql_2)

    # 判断距离上次执行的这段时间内，源表中是否有新数据生成
    if len(raw_datas)<1:
        return start_id,0  # 如果无新数据，直接返回不执行分析，最新id仍为上次取到的最大id

    end_id=raw_datas[-1][1]  # 最后一条分析的数据id

    # 这一批数据保存为 DataFrame
    data_df = pd.DataFrame(data=[], columns=["contnet_id","id", "text","gmt_crawler"])  # text列包括标题和内容
    for i in range(0, len(raw_datas)):
        # 取需要的字段
        data_df.loc[i] = [raw_datas[i][0],raw_datas[i][1], (raw_datas[i][2] + "" + raw_datas[i][3]),raw_datas[i][4]]

    """分词处理"""
    # 对每一行（每篇文章进行分词操作后,添加到新列‘segment_text’中）
    data_df['segment_text'] = data_df['text'].apply(jieba_process)
    # 除去无用样本
    data_df = data_df[~data_df['segment_text'].isin(['无用'])]

    """批量分析（预测+保存结果）"""
    predict(model_name,data_df)

    return end_id,data_df.shape[0]



















