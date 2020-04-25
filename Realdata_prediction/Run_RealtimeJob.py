"""
定时任务程序（测试）
"""
import arrow
#from batch_predict import dataObj_149,dataObj_150,iter_run,update_lastId
from batch_predict_v2 import dataObj_149,dataObj_150,iter_run,update_lastId
from apscheduler.schedulers.blocking import BlockingScheduler
import logging
import random

last_id=0
multi_num=4
current_month=7
orgData_table = ""
target_table = ""


def run_realtime_data():
    """
    跑实时数据
    """
    global last_id

    start_id=last_id
    instance_id = random.randint(0, 10000)  # 随机生成一个 instance 的id，作为其唯一标识

    start_time=arrow.now()
    print(" **** Schedule job instance NO.{} started at{}; Beginning id:{}".format(instance_id,start_time,last_id))

    # 更新 last_id ，用本次取到的最大id（如果前一个实例未执行完下一个就开始，保证下一个实例的id不会重复取）
    last_id=update_lastId(orgData_table,last_id)

    # 从上一个实例取到的最大id开始取，对目前为止所有的新数据做分析，并取到分析过的最大id
    #end_id, data_num = iter_run(orgData_table, model_name, start_id, multi_num, target_table)  对应于 batch_predict
    end_id, data_num = iter_run(orgData_table,target_table,start_id,multi_num)   # 对应于 batch_predict_v2

    print(" *** Schedule job instance NO.{} finished.using time: {}.  {} datas analysed! End id:{}\n".format(instance_id,(arrow.now()-start_time),data_num,end_id))


if __name__=='__main__':
    """
    程序开始后，初始化 last_id
    ## 若分析结果表为空（首次运行），就用源表的当前最新一条数据id
    ## 若之前已有分析结果（程序中断），则用分析结果表的最新数据id
    """

    sql="select count(*) from "+target_table
    ana_num=dataObj_149.data_read(sql)[0][0]
    if ana_num<1:
        sql_1 = "SELECT id FROM " +orgData_table+ " ORDER BY id desc limit 1"
        last_id = dataObj_150.data_read(sql_1)[0][0]
    else:
        sql_2 = "SELECT id FROM " + target_table + " ORDER BY id desc limit 1"
        last_id = dataObj_149.data_read(sql_2)[0][0]

    """
    设置实时分析的定时任务（每15s执行一次)
    """
    schedule = BlockingScheduler()
    schedule.add_job(run_realtime_data, 'interval',seconds=10,max_instances=200)
    schedule.start()






