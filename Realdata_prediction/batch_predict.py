from multiprocessing import Process
import pymysql
import pandas as pd
import numpy as np
import arrow
import re
from sklearn.externals import joblib
from dataObj import DataObj
from data_process import jieba_process,pre_process


dataObj_1 = DataObj("", "", "", "")
dataObj_2 = DataObj("", "", "", "")


def predict(model,data,sql_conn,target_table):
    """
    单条预测
    :param model: 加载后的模型对象
    :param data:  单条文本,DataFrame中的一行，包括分分词后的文本
    :param sql_conn: pymysql.connect() 创建的数据库连接对象
    :return: 无返回值，预测结果直接插入结果表
    """
    text_vector = pre_process(data['segment_text'])  # 文档向量
    if text_vector == "zero":  # 如果文本所有词语都未出现在预训练词库中，直接判为0
        prediction = 0
    else:
        prediction = model.predict([text_vector])  # 否则，输入文档向量，用模型预测
    try:
        # 预测结果保存到数据表 ,保留源表中所有字段，在分类字段填入预测类型（0或1）
        sql = "insert into "+target_table+" (content_id,id,text,time,Tag) values (%s, %s,'%s','%s',%s)"
        curs = sql_conn.cursor()
        curs.execute(sql % (int(data['content_id']), int(data['id']), pymysql.escape_string(data['text']),
                            str(data['gmt_crawler']), int(prediction[0])))
        sql_conn.commit()
    except Exception as e:
        # 记录下未分析成功的数据id，和错误原因
        print("Insert error:{}.  Data id:{} analysis failed".format(e, data['id']))


def batch_segment(data_df):
    """
    批量分词
    :param data_df: 未分词数据
    :return:  分词处理后的数据（加入了新列‘segment_text’）
    """
    stime=arrow.now()

    # 对每一行（每篇文章进行分词操作后,添加到新列‘segment_text’中）
    data_df['segment_text'] = data_df['text'].apply(jieba_process)
    # 除去无用样本
    data_df = data_df[~data_df['segment_text'].isin(['无用'])]

    print("Child process of batch segment running on {} datas, using time{}".format(data_df.shape[0],arrow.now()-stime))

    return data_df


def serial_predict(model,raw_data,target_table):
    """
    单进程批量预测（串行）
    :param model:  加载后的模型对象
    :param raw_data:  原始数据（未分词） DataFrame
    """
    # 批量分词处理
    segment_df=batch_segment(raw_data)

    startTime = arrow.now()
    # 创建数据库连接（每批数据只需要创建一次）
    conn = pymysql.connect(host="", user="", passwd="", db="",
                           charset='utf8')

    for n in range(segment_df.shape[0]):
        data=segment_df.iloc[n] # 取出每条数据
        predict(model,data,conn,target_table)

    conn.close()

    print("Child process of batch prediction running on {} datas, using time{}".format(raw_data.shape[0],arrow.now()-startTime))


def multi_predict(model,raw_data,multi_num,target_table):
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
        p = Process(target=serial_predict, args=(model, sub_df,target_table))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


"""测试用：一次性跑一整批历史数据"""
def history_run(orgData_table,target_table,start_id,model_name,multi_num):
    start_time = arrow.now()
    print("Program started at {}".format(start_time))

    # 加载模型
    model = joblib.load(model_name)

    sql_2 = "select content_id,id,title,content,time from " + orgData_table + " where id> " + str(start_id)+" limit 50"
    results = dataObj_1.data_read(sql_2)

    raw_datas=[ [data[0],data[1],data[2]+data[3],data[4]] for data in results]  # 合并标题和内容

    """批量分析（预测+保存结果）"""
    # 并行跑
    multi_predict(model, raw_datas, multi_num,target_table)

    print("Running on {} datas, using time{}".format(len(raw_datas), arrow.now() - start_time))


def update_lastId(orgData_table,start_id):
    """
    更新最大id
    :param orgData_table:
    :param start_id:
    :return:
    """
    sql = "select id from " + orgData_table + " where id> " + str(start_id)
    ids = dataObj_1.data_read(sql)

    # 若无新数据，返回原来的start_id (last_id不变）
    if len(ids)<1:
        return start_id

    # 有新数据，返回最后一条的id，用其更新last_id
    return ids[-1][0]


def iter_run(orgData_table,model_name,start_id,multi_num,target_table):
    """
    :param orgData_table:   源数据表名
    :param model_name:     预测用的ML模型（pkl文件名）
    :param start_id:      上次取到的最新数据id，本次分析从last_id+1开始取到当前最新数据
    :return:       分析完后最后一条的id，分析的数据条数
    """
    # 加载模型
    model = joblib.load(model_name)

    """取批量数据（从上次的last_id到目前的所有新数据）"""
    sql_2 = "select content_id,id,title,content,gmt_crawler from "+orgData_table+" where id> "+str(start_id)
    results = dataObj_2.data_read(sql_2)

    # 判断距离上次执行的这段时间内，源表中是否有新数据生成
    if len(results)<1:
        return start_id,0  # 如果无新数据，直接返回不执行分析，最新id仍为上次取到的最大id

    end_id=results[-1][1]  # 最后一条分析的数据id

    raw_datas = [[data[0], data[1], data[2] + data[3], data[4]] for data in results]  # 合并标题和内容


    """批量分析（预测+保存结果）"""
    # 并行多进程跑
    if len(raw_datas)<multi_num:
        # 如果本次数据少于进程数量，就给每个数据单独分配一个进程
        multi_predict(model, raw_datas, len(raw_datas),target_table)
    else:
        # 如果数据量大于进程数，所有数据在各进程上平均分配
        multi_predict(model, raw_datas, multi_num,target_table)

    return end_id,len(raw_datas)


if __name__=="__main__":
    orgData_table = ""
    target_table = ""
    start_id = 351
    multi_num = 4
    model_name = ""

    #history_run(orgData_table,target_table,start_id,model_name,multi_num)


