import pymysql
import pandas as pd
import numpy as np
import arrow
import re
from sklearn.externals import joblib
from dataObj import DataObj
from model_predict import jieba_process,pre_process

dataObj_150 = DataObj("192.168.20.150", "root", "les-admin-cx", "les_crawler_data_analysis")
dataObj_149 = DataObj("192.168.20.149", "root", "admin123!@#", "qh_Realtime_Test")


"""单进程批量预测"""
def predict(model_name,raw_data):
    startTime0 = arrow.now()

    # 加载模型
    model = joblib.load(model_name)
    conn = pymysql.connect(host="192.168.20.149", user="root", passwd="admin123!@#", db="qh_Realtime_Test",
                           charset='utf8')
    curs = conn.cursor()

    # 逐个遍历，单条预测
    for n in range(raw_data.shape[0]):
        try:
            data = raw_data.iloc[n]
        except Exception as e:
            print(e)
            continue
        text_vector = pre_process(data['segment_text'])  # 文档向量
        if text_vector == "zero":  # 如果文本所有词语都未出现在预训练词库中，直接判为0
            prediction = 0
        else:
            prediction = model.predict([text_vector])  # 否则，输入文档向量，用模型预测
        try:
            # 预测结果保存到数据表 ,保留源表中所有字段，在分类字段填入预测类型（0或1）
            sql = "insert into temp_GBDT_Predict (id,content_id,text,gmt_crawler,Tag,process_type) values (%s, %s,'%s','%s',%s,'%s')"
            curs.execute(sql % (int(data['id']),int(data['contnet_id']), pymysql.escape_string(data['text']),
                                str(data['gmt_crawler']), int(prediction[0]),"single"))  # 单进程标识
            conn.commit()
        except Exception as e:
            # 记录下未分析成功的数据id，和错误原因
            print("Insert error:{}.  Data id:{} analysis failed".format(e, data['id']))

    conn.close()


"""多进程批量预测"""
def multi_predict(model_name,raw_data):
    pass


if __name__=="__main__":
    pass