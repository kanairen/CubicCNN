# coding:utf-8

import time
import functools
from src.util.date import ymdt

__author__ = 'ren'


def client(func):
    """
    Client関数・メソッドが呼び出された時の処理
    :param func: 呼び出し関数
    :return: デコレータ関数
    """

    @functools.wraps(func)
    def timer_func(*args, **kwargs):
        print "{} Run Client...".format(ymdt())
        start = time.clock()

        func(*args, **kwargs)

        stop = time.clock()
        print "process time : ", stop - start, " s"

    return timer_func
