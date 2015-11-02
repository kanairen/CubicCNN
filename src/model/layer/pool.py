# coding:utf-8

import numpy as np
from theano import config, tensor as T, scan

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

        # 各入力の各プーリング領域における最大値を取得する
        result, update = scan(fn=lambda input, W: T.max(input * W, axis=1),
                              sequences=[inputs_symbol],
                              non_sequences=self.W)
        return result
