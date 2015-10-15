# coding:utf-8

import time
from src.util.config import GPU_ID
from chainer import cuda

__author__ = 'ren'


def client(func):
    print "Run Client..."

    # use gpu
    cuda.get_device(GPU_ID).use()

    def timer_func(f):
        start = time.clock()
        f()
        stop = time.clock()
        print "process time : ", stop - start, " s"

    return lambda : timer_func(func)
