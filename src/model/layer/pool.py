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

    def __init__(self, img_size, in_channel, k_size, stride=None, T=None,
                 h=None, pad=0, dtype=config.floatX, activation=relu,
                 pool_type=POOL_MAX):

        # フィルタサイズ
        kw, kh = pair(k_size)

        # 実行するプーリングの種類
        self.pool_type = pool_type

        if stride is None:
            stride = k_size

        # フィルタベクトル
        if h is None:
            h = np.ones((in_channel, in_channel, kh, kw), dtype=dtype)
        self.h = shared(h, name='h', borrow=True)

        super(PoolLayer, self).__init__(img_size, in_channel, in_channel,
                                        k_size, stride, T, None, True, h,
                                        dtype, activation)

    def update(self, cost, learning_rate=0.001):
        return None

    def output(self, inputs_symbol):
        if self.pool_type == PoolLayer.POOL_MAX:
            output = self.max_pooling(inputs_symbol)
        elif self.pool_type == PoolLayer.POOL_AVERAGE:
            output = self.averate_pooling(inputs_symbol)
        else:
            raise RuntimeError("pool_type is invalid.")

        return self.activation(output)

    def max_pooling(self, inputs_symbol):

        # データ数
        n = inputs_symbol.shape[0]

        # 一時的に4次元テンソルの形に
        inputs_4d = T.reshape(inputs_symbol,
                              (n, self.in_channel, self.img_h, self.img_w))

        # 畳み込み対象となる画素のみを抽出したテンソル
        col = self.im2col(inputs_4d)

        reshape_col = T.reshape(col, (
            n, self.in_channel, self.kh * self.kw, self.out_h, self.out_w))

        max_col = T.flatten(T.max(reshape_col, axis=2), outdim=2)

        return max_col
