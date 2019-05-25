import os
from multiprocessing import Process
import random


# 对大循环里每个数字执行的操作
def work_func(num):
    print("I got a number{}".format(num))


# 大循环，遍历20个数字
# 把这个大循环分到四个进程上跑（考虑不能整除的问题）
def loop():
    number_lst=[ random.randint(0,100) for i in range(20)]
    for num in number_lst:
        work_func(num)

