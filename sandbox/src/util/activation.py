# coding=utf-8

import numpy as  np

__author__ = 'kanairen'


def sigmoid(x):
    return 1. / (1 + np.exp(x))


def d_sigmoid(x):
    return x * (1. - x)


def relu(x):
    return x * (x > 0)


def d_relu(x):
    return 1. * (x > 0)


def softmax(x):
    exp = np.exp(x - np.max(x))
    if exp.ndim == 1:
        return exp / np.sum(exp, axis=0)
    else:
        return exp / np.array([np.sum(exp, axis=1)]).T
