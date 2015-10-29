# coding:utf-8

import numpy as np
from theano import config, tensor as T

from src.model.layer.convolution import ConvLayer2d
from src.util.activation import relu
from src.util.conv import pair

__author__ = 'ren'


class PoolLayer(ConvLayer2d):
    POOL_MAX = 0
    POOL_AVERAGE = 1

    def __init__(self, img_size, in_channel, k_size, pad=0, dtype=config.floatX,
                 activation=relu, pool_type=POOL_MAX):

        kw, kh = pair(k_size)

        # フィルタベクトル
        filter = np.ones((in_channel * in_channel * kh * kw), dtype=dtype)

        super(PoolLayer, self).__init__(img_size, in_channel, in_channel,
                                        k_size, k_size, pad, None, None, filter,
                                        dtype, activation)

        self.pool_type = pool_type

    def update(self, cost, learning_rate=0.01):
        return []

    def output(self, inputs_symbol):
        self.filtering()

        linear = T.dot(inputs_symbol, self.W.T) + self.b
        if self.pool_type == PoolLayer.POOL_MAX:
            return T.max(linear, axis=1)
        elif self.pool_type == PoolLayer.POOL_AVERAGE:
            return T.mean(linear, axis=1)
        else:
            raise RuntimeError("pool_type is invalid.")
