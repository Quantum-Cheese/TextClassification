"""批量多进程测试，依次取批量数据，每批数据多进程并行跑"""
import arrow
from batch_predict_v2 import dataObj_149,dataObj_150,parallel_predict
import warnings
warnings.filterwarnings("ignore", category=Warning)


def batch_run(orgData_table,target_table,start_id,multi_num,batch_size):
    """
    :param orgData_table: 数据来源表
    :param target_table:  结果表
    :param start_id:   本批的起始数据id
    :param multi_num:  多进程数量
    :param batch_size:  本批数据量
    """
    start_time = arrow.now()
    print("Program started at {}".format(start_time))

    sql_2 = "select content_id,id,title,content,time from " + orgData_table + " where id> " + str(start_id) + " limit " + str(batch_size)
    results = dataObj_1.data_read(sql_2)

    raw_datas = [[data[0], data[1],data[2]+data[3],data[4]] for data in results]  # 合并标题和内容

    """批量分析（预测+保存结果）"""
    # 并行跑
    parallel_predict(raw_datas, multi_num, target_table)

    print("Running on {} datas, using time{}".format(len(raw_datas), arrow.now() - start_time))


def historyJob(org_table,target_table,multi_num,batch_size):
    start_time = arrow.now()

    last_id=0

    while True:
        # 检查数据是否取完
        sql1="select count(id) from "+str(org_table) + " where id>"+ str(last_id)
        remain_num=dataObj_1.data_read(sql1)[0][0]
        if remain_num==0:
            break

        # 取目标结果表的最大id
        sql2="select count(id) from "+ str(target_table)
        target_num=dataObj_2.data_read(sql2)[0][0]
        if target_num>0:
            sql = "select max(id) from " + str(target_table)
            last_id = dataObj_2.data_read(sql)[0][0]

        batch_run(orgData_table,target_table,last_id,multi_num,batch_size)

    print("Program finished, running time {}".format(arrow.now()-start_time))


if __name__=="__main__":
    orgData_table = ""
    multi_num = 4
    target_table = ""
    batch_size=400

    historyJob(orgData_table,target_table,multi_num,batch_size)





