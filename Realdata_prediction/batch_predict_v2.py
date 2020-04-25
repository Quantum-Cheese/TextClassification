from multiprocessing import Process,Queue
import pymysql
import pandas as pd
import numpy as np
import arrow
import re
from sklearn.externals import joblib
from dataObj import DataObj
from data_process import jieba_process,pre_process
import warnings
warnings.filterwarnings("ignore", category=Warning)

dataObj_1 = DataObj("", "", "", "")
dataObj_2 = DataObj("", "", "", "")

model_name = ""
model=joblib.load(model_name)


def batch_segment(data_df):
    """
    批量分词
    :param data_df: 未分词数据
    :return:  分词处理后的数据（加入了新列‘segment_text’）
    """
    stime = arrow.now()

    # 对每一行（每篇文章进行分词操作后,添加到新列‘segment_text’中）
    data_df['segment_text'] = data_df['text'].apply(jieba_process)
    # 除去无用样本
    data_df = data_df[~data_df['segment_text'].isin(['无用'])]

    print("Child process of batch segment running on {} datas, using time{}".format(data_df.shape[0],arrow.now()-stime))

    return data_df


"""单条文本预测"""
def predict(text):
    """
    :param text: 分词后的文本
    :return: 预测结果（0或1）
    """
    # 文本转化为文档向量
    text_vector = pre_process(text)
    if text_vector == "zero":  # 如果文本所有词语都未出现在预训练词库中，直接判为0
        prediction = 0
    else:
        prediction = model.predict([text_vector])  # 否则，输入文档向量，用模型预测
    return prediction


"""批量预测子进程"""
def predict_child_process(data_df,target_table):

    # 批量分词
    data_df=batch_segment(data_df)

    # 批量预测
    start1=arrow.now()
    data_df['tag'] = data_df['segment_text'].apply(predict)
    print("Batch prediction on {} datas,using time{}".format(data_df.shape[0],arrow.now()-start1))

    # 保存：顺序遍历
    start2=arrow.now()
    conn = pymysql.connect(host="192.168.20.149", user="root", passwd="admin123!@#", db="qh_Realtime_Test",charset='utf8')
    for n in range(data_df.shape[0]):
        data=data_df.iloc[n]
        try:
            # 预测结果保存到数据表 ,保留源表中所有字段，在分类字段填入预测类型（0或1）
            sql = "insert into " + target_table +" (id,content_id,text,gmt_crawler,Tag) values (%s, %s,'%s','%s',%s)"
            curs = conn.cursor()
            curs.execute(sql % (int(data['id']), int(data['content_id']), pymysql.escape_string(data['text']),
                                str(data['gmt_crawler']), int(data['tag'])))  # 单多进程标识（并行）
            conn.commit()
        except Exception as e:
            # 记录下未分析成功的数据id，和错误原因
            print("Insert error:{}.  Data id:{} analysis failed".format(e, data['id']))

    print("saving {} datas,using time{}".format(data_df.shape[0],arrow.now()-start2))


"""多进程预测（把一大批数据分成小batch并行跑)"""
def parallel_predict(raw_data,multi_num,target_table):
    interval=len(raw_data)//multi_num
    indss = []
    # 先把整除的部分平均分
    for j in range(multi_num):
        inds = list(range(j * interval, (j + 1) * interval))
        indss.append(inds)
    # 再把余数从前到后分到各小批里，分完为止
    for i in range(len(raw_data) % multi_num):
        indss[i].append(interval * multi_num + i)

    processes = []
    for inds in indss:
        data_lst=[list(raw_data[ind]) for ind in inds]
        sub_df=pd.DataFrame(data=data_lst,columns=["content_id", "id", "text", "time"])
        # 每个小batch的数据分给一个单独的子进程
        p = Process(target=predict_child_process, args=(sub_df,target_table))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


"""一次性跑整批历史数据"""
def history_run(orgData_table,target_table,start_id,multi_num,batch_size):
    start_time=arrow.now()
    print("Program started at {}".format(start_time))

    sql_2 = "select content_id,id,title,content,time from " + orgData_table + " where id> " + str(start_id)+ " limit "+batch_size
    results = dataObj_1.data_read(sql_2)

    raw_datas = [[data[0], data[1], data[2] + data[3], data[4]] for data in results]  # 合并标题和内容

    """批量分析（预测+保存结果）"""
    # 并行跑
    parallel_predict(raw_datas, multi_num, target_table)

    print("Running on {} datas, using time{}".format(len(raw_datas),arrow.now()-start_time))


def update_lastId(orgData_table,start_id):

    """取批量数据（从上次的last_id到目前的所有新数据）"""
    sql = "select id from " + orgData_table + " where id> " + str(start_id)
    ids = dataObj_1.data_read(sql)

    # 若无新数据，返回原来的start_id (last_id不变）
    if len(ids)<1:
        return start_id

    # 有新数据，返回最后一条的id，用其更新last_id
    return ids[-1][0]


def iter_run(orgData_table,target_table,start_id,multi_num):
    """
    :param orgData_table:
    :param model_name:
    :param start_id:
    :return:
    """
    """取批量数据（从上次的last_id到目前的所有新数据）"""
    sql_2 = "select content_id,id,title,content,time from " + orgData_table + " where id> " + str(start_id)
    results = dataObj_1.data_read(sql_2)

    # 判断距离上次执行的这段时间内，源表中是否有新数据生成
    if len(results) < 1:
        return start_id, 0  # 如果无新数据，直接返回不执行分析，最新id仍为上次取到的最大id

    end_id = results[-1][1]  # 最后一条分析的数据id

    raw_datas = [[data[0], data[1], data[2] + data[3], data[4]] for data in results]  # 合并标题和内容

    """批量分析（预测+保存结果）"""
    # 并行多进程跑
    if len(raw_datas) < multi_num:
        # 如果本次数据少于进程数量，就给每个数据单独分配一个进程
        parallel_predict(raw_datas, len(raw_datas), target_table)
    else:
        # 如果数据量大于进程数，所有数据在各进程上平均分配
        parallel_predict(raw_datas, multi_num, target_table)

    return end_id, len(raw_datas)


if __name__=="__main__":
    pass

