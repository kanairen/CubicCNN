# coding:utf-8

import datetime

__author__ = 'ren'

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def ymdt():
    return datetime.datetime.today().strftime(TIME_FORMAT)