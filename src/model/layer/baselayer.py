# coding:utf-8

import os
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from src.helper.config import path_res_numpy_weights, path_res_numpy_biases
from src.util.io import numpy_save

__author__ = 'ren'


class BaseLayer(object):
    def __init__(self, is_dropout, dropout_rate):
        seed = 1111

        # 重み・バイアスパラメタ
        self.W = None
        self.b = None

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

    def save_weights(self, file_name):
        numpy_save(os.path.join(path_res_numpy_weights, file_name), self.W)

    def save_biases(self, file_name):
        numpy_save(os.path.join(path_res_numpy_biases, file_name), self.b)

    def __str__(self):
        return "[{:^13}]".format(self.__class__.__name__)
