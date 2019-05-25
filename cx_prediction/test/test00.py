from apscheduler.schedulers.blocking import BlockingScheduler
import random
import arrow
import time
import datetime
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT)


def func():
    start_time=arrow.now()
    id=random.randint(0,10000)
    print("NO. {} job started".format(id))

    time.sleep(5)
    print("I'm still running!!!")
    print("NO. {} job finised\n Using time : {}\n".format(id,arrow.now()-start_time))




if __name__=="__main__":
    # schedule = BlockingScheduler()
    # schedule.add_job(func, 'interval', seconds=2,max_instances=10)  # 设置同时执行的最大任务数
    # print("Program started at{}".format(arrow.now()))
    # schedule.start()
    i=0
    try:
        print(2/i)
    except Exception as e:
        print(e)
    finally:
        logging.error("Wrong number:{}".format(i))

    print(type(2019050262044))






