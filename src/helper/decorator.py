# coding:utf-8

import time

__author__ = 'ren'


def client(func):
    def timer_func(**kwargs):
        print "Run Client..."
        start = time.clock()
        func(**kwargs)
        stop = time.clock()
        print "process time : ", stop - start, " s"

    return timer_func
