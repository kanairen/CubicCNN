# coding:utf-8

import time
from src.util.date import ymdt

__author__ = 'ren'


def client(func):
    def timer_func(**kwargs):
        print "{} Run Client...".format(ymdt())
        start = time.clock()
        func(**kwargs)
        stop = time.clock()
        print "process time : ", stop - start, " s"

    return timer_func
