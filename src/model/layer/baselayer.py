# coding:utf-8

import os
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from src.util.io import numpy_save

__author__ = 'ren'


class BaseLayer(object):
    def __init__(self, is_dropout, dropout_rate):
        seed = 1111

        # 乱数生成器
        self.rnd = np.random.RandomState(seed)

        # Theanoシンボル計算で用いる乱数生成器
        self.srnd = RandomStreams(seed)

        # ドロップアウト
        self.is_dropout = is_dropout

        # ドロップアウトされるユニットの割合
        self.dropout_rate = dropout_rate

        # 学習フラグ
        self.is_train = False

    def update(self, cost, learning_rate):
        raise NotImplementedError(
            "layer objects should be implemented 'update'.")

    def output(self, inputs_symbol):
        raise NotImplementedError(
            "layer objects should be implemented 'output'.")

    def __str__(self):
        return "[{:^13}]".format(self.__class__.__name__)
