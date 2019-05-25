"""
从数据库取出数据后直接进行预处理并预测，预测结果放入数据库的新表中，不经过csv文件的存储
"""

import pymysql
import pandas as pd
import numpy as np
import arrow
from sklearn.externals import joblib
from dataObj import DataObj
from data_preprocess_v3 import segment
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import warnings
warnings.filterwarnings('ignore')
from data_preprocess_v3 import jieba_process


"""数据预处理：把单条文本转换为文档向量"""
def pre_process(text):
    startTime=arrow.now()

    #读取词权重词典
    word_weight_1 = pd.read_csv('datas/word_weight_0515.csv')
    word_weight_1.columns = ['word', 'weight']

    # 读取word2vec模型
    model = word2vec.Word2Vec.load('datas/word2vec/w2v_3.model')
    #model = KeyedVectors.load_word2vec_format("datas/word2vec/vector_word_cnn.txt")  # 从词向量文件加载word2vec
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


"""加载模型预测一批数据（从参数传过来的 DataFrame），并把结果保存到数据库"""
def predict(model_name,raw_data):

    # 加载模型
    model = joblib.load(model_name)
    conn = pymysql.connect(host="192.168.20.149", user="root", passwd="admin123!@#", db="text_classification_samples",
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
            # 预测结果保存到数据表 qh_testResults_1
            sql = "insert into qh_testResults_2 (id,text,tag) values (%s,'%s',%s)"
            curs.execute(sql % (int(data['org_id']), pymysql.escape_string(data['text']), int(prediction[0])))
            conn.commit()
        except Exception as e:
            print(str(e))

        #print("--Single run completed, using time{}".format(arrow.now()-startTime0))

    conn.close()



"""从数据库读取批量数据并预测，控制每次取得的数据条数"""
def iter_run(model_name,batch_size):
    # 读取数据
    url = "192.168.20.149"
    username = "root"
    password = "admin123!@#"
    db = "text_classification_samples"
    data_obj = DataObj(url, username, password, db)

    last_id=2019030478831
    while True:
        startTime=arrow.now()
        sql = "select org_id,title,content from real_data_test where crawler_time < '2019-03-21 00:00:00'" \
            + "and org_id> "+ str(last_id) +" limit "+str(batch_size)
        last_id+=batch_size
        raw_datas = data_obj.data_read(sql)
        if len(raw_datas)==0:
            print("Program finished!")
            break
        # 这一批数据保存为 DataFrame
        data_df = pd.DataFrame(data=[], columns=["org_id", "text"])  # text列包括标题和内容
        for i in range(0, len(raw_datas)):
            data_df.loc[i] = [raw_datas[i][0], raw_datas[i][1] + "" + raw_datas[i][2]]  # 合并标题和内容

        # 对每一行（每篇文章进行分词操作后,添加到新列‘segment_text’中）
        data_df['segment_text'] = data_df['text'].apply(jieba_process)
        # 除去无用样本
        data_df = data_df[~data_df['segment_text'].isin(['无用'])]

        # 对这批数据进行预测，结果插入 database
        predict(model_name,data_df)

        print("Batch run completed! batch size:{} ; using time:{}".format(batch_size,arrow.now()-startTime))



if __name__=="__main__":

    iter_run("pkls/GBDT_0515.pkl",10)







