# coding:utf-8

import numpy as np
from theano import config, tensor as T, scan, shared

from filterlayer import FilterLayer
from src.util.activation import relu
from src.util.sequence import pair

__author__ = 'ren'


class PoolLayer(FilterLayer):
    POOL_MAX = 0
    POOL_AVERAGE = 1

    def __init__(self, img_size, in_channel, k_size, T=None, h=None, pad=0,
                 dtype=config.floatX, activation=relu, pool_type=POOL_MAX):

        # フィルタサイズ
        kw, kh = pair(k_size)

        # 実行するプーリングの種類
        self.pool_type = pool_type

        # フィルタベクトル
        if h is None:
            h = np.ones((in_channel * in_channel * kh * kw), dtype=dtype)
        self.h = shared(h, name='filter', borrow=True)

        super(PoolLayer, self).__init__(img_size, in_channel, in_channel,
                                        k_size, k_size, T, None, True, h,
                                        dtype, activation)

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

        W = T.tensordot(self.h, self.T, axes=(0, 2))

        # 各入力の各プーリング領域における最大値を取得する
        result, update = scan(fn=lambda input, W: T.max(input * W, axis=1),
                              sequences=[inputs_symbol],
                              non_sequences=W)
        return result
