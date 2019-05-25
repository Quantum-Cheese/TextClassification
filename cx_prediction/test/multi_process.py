import multiprocessing
import random
from multiprocessing import Pool
from threading import Thread
import os
from multiprocessing import Process


def loop():
    while True:
        pass


def multi_thread():
    for i in range(4):
        t = Thread(target=loop)
        t.start()

    while True:
        pass


def multi_process():
    for i in range(4):
        t = Process(target=loop)
        t.start()

    while True:
        pass


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())  # 获取父进程的pid
    print('process id:', os.getpid())    # 获取子进程的pid


def f(name):
    info('function f')
    print('hello', name)
    loop()



if __name__ == '__main__':

    # 多线程和多进程对比
    #multi_thread()
    #multi_process()


    # 创建一个进程实例，该进程执行函数 f , 并给 f 传入实参
    # p = Process(target=f, args=('bob',))
    # p.start()
    # p.join()

    # 创建四个进程，每个进程都执行相同的函数 f ,传入不同的实参
    for i in range(100):
        p = Process(target=f, args=('child process '+str(random.randint(0,50)),))
        p.start()


















