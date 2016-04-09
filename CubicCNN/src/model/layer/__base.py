# coding: utf-8

import numpy as np
from theano import config
from theano.tensor.shared_randomstreams import RandomStreams


class BaseLayer(object):
    def __init__(self, layer_id, n_in, n_out, activation, is_dropout,
                 dropout_rate):
        # レイヤオブジェクトに割り振るID
        self.layer_id = layer_id

        # 入出力ユニット数
        self.n_in = n_in
        self.n_out = n_out

        # 活性化関数
        self.activation = activation

        # 学習パラメタ
        self.params = None

        # dropoutを有効にするかどうか
        self.is_dropout = is_dropout

        self.dropout_rate = dropout_rate

        self.rnd = np.random.RandomState(1111)

        self.trnd = RandomStreams(1111)

    def output(self, input, is_train):
        # dropoutの変更を反映させるため、出力を動的に作る
        raise NotImplementedError

    def _activate(self, u, is_train, dtype=config.floatX):
        z = self.activation(u)
        if self.is_dropout:
            if is_train:
                z *= self.trnd.binomial(size=z.shape, p=0.5, dtype=dtype)
            else:
                z *= self.dropout_rate
        return z

    def __str__(self):
        return "[{}:{:^13}]".format(self.layer_id, self.__class__.__name__)
