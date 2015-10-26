# coding=utf-8

import functools
import operator

__author__ = 'kanairen'


def product(x):
    """
    受け取ったリストの総乗を返す
    :param x:
    :return:
    """
    prod = functools.partial(functools.reduce, operator.mul)
    return prod(x)

