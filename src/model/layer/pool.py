# coding:utf-8

import six
import numpy as np
from theano import config, tensor as T, scan, printing

from src.model.layer.conv import ConvLayer2d
from src.util.activation import relu
from src.util.sequence import pair

__author__ = 'ren'


class PoolLayer(ConvLayer2d):
    POOL_MAX = 0
    POOL_AVERAGE = 1

    def __init__(self, img_size, in_channel, k_size, pad=0, dtype=config.floatX,
                 activation=relu, pool_type=POOL_MAX):

        # フィルタサイズ
        kw, kh = pair(k_size)

        # フィルタベクトル
        filter = np.ones((in_channel * in_channel * kh * kw), dtype=dtype)

        super(PoolLayer, self).__init__(img_size, in_channel, in_channel,
                                        k_size, k_size, pad, None, None, True,
                                        filter, dtype, activation)

        # 実行するプーリングの種類
        self.pool_type = pool_type

    def update(self, cost, learning_rate=0.01):
        return None

    def output(self, inputs_symbol):
        if self.pool_type == PoolLayer.POOL_MAX:
            return self.max_pooling(inputs_symbol)
        elif self.pool_type == PoolLayer.POOL_AVERAGE:
            return self.averate_pooling(inputs_symbol)
        else:
            raise RuntimeError("pool_type is invalid.")

    def max_pooling(self, inputs_symbol):
        # Wを初期化
        self.init_weight()
        # 重み線型結合
        linear = T.dot(inputs_symbol, self.W.T)

        max_args = T.argmax(linear)




    @staticmethod
    def max_pool_weighting(v, w_row, weight):
        max_arg = T.argmax(v)
        zeros = T.zeros_like(v)
        weight[w_row].fill(0)

        return T.set_subtensor(zeros[max_arg], v[max_arg])
