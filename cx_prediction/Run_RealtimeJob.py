"""
定时任务程序（测试）
"""
import arrow
from model_predict import dataObj_149,dataObj_150,iter_run,update_lastId
from apscheduler.schedulers.blocking import BlockingScheduler
import logging
import random

model_name = "file/GBDT_0515.pkl"
last_id=0

orgData_table="les_crawler_ana"+"_201905"  # todo：换月更新

def run_realtime_data():
    """
    跑实时数据，跟现有分析程序同步
    :return:
    """
    global last_id

    start_id=last_id  # 取分析数据的起始id

    instance_id = random.randint(0, 10000)  # 随机生成一个 instance 的id，作为其唯一标识

    start_time=arrow.now()
    logging.debug("\n ****** Schedule job instance NO.{} started at{}; Beginning id:{}".format(instance_id,start_time,last_id))
    print(" **** Schedule job instance NO.{} started at{}; Beginning id:{}".format(instance_id,start_time,last_id))

    # 更新 last_id ，用本次取到的最大id（如果前一个实例未执行完下一个就开始，保证下一个实例的id不会重复取）
    last_id=update_lastId(orgData_table,last_id)

    # 从上一个实例取到的最大id开始取，对目前为止所有的新数据做分析，并取到分析过的最大id
    end_id, data_num = iter_run(orgData_table,model_name, start_id)

    logging.info("\n ****** Schedule job instance NO.{} finished.using time: {}.  {} data analysed! End id:{}\n".format(instance_id,(arrow.now()-start_time),data_num,end_id))

    print(" *** Schedule job instance NO.{} finished.using time: {}.  {} data analysed! End id:{}\n".format(instance_id,(arrow.now()-start_time),data_num,end_id))


if __name__=='__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename='log/real_time_running.log', level=logging.INFO, format=LOG_FORMAT)

    """
    程序开始后，初始化 last_id
    ## 若分析结果表为空（首次运行），就用源表的当前最新一条数据id
    ## 若之前已有分析结果（程序中断），则用分析结果表的最新数据id
    """
    sql="select count(*) from GBDT_Predict_realtime_1"
    ana_num=dataObj_149.data_read(sql)[0][0]
    if ana_num<1:
        sql_1 = "SELECT id FROM `les_crawler_ana_201905` ORDER BY id desc limit 1"
        last_id = dataObj_150.data_read(sql_1)[0][0]
    else:
        sql_2="SELECT id FROM `GBDT_Predict_realtime_1` ORDER BY id desc limit 1"
        last_id = dataObj_149.data_read(sql_2)[0][0]

    ## todo：如果程序中断，但结果表的数据id不连续，直接取最大id会漏掉一些未分析数据，如何处理？


    """
    设置实时分析的定时任务（每15s执行一次)
    """
    schedule = BlockingScheduler()
    schedule.add_job(run_realtime_data, 'interval',seconds=15,max_instances=200)
    schedule.start()






