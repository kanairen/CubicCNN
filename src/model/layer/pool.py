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

    def __init__(self, img_size, in_channel, k_size, T=None, filter=None, pad=0,
                 dtype=config.floatX, activation=relu, pool_type=POOL_MAX):

        super(PoolLayer, self).__init__()

        # 画像サイズ
        img_w, img_h = pair(img_size)

        # フィルタサイズ
        kw, kh = pair(k_size)

        # ストライド（フィルタサイズと同じ）
        sw, sh = pair(k_size)

        # 入力・出力ユニット数
        n_in = img_w * img_h * in_channel
        n_out = img_w * img_h * in_channel / (sw * sh)

        # 重み・フィルタ変換行列
        if T is None:
            T = self.init_T(img_w, img_h, kw, kh, sw, sh, n_in, n_out,
                            in_channel, in_channel, dtype=dtype)
        self.T = shared(T, name='T', borrow=True)

        # フィルタベクトル
        if filter is None:
            filter = np.ones((in_channel * in_channel * kh * kw), dtype=dtype)
        self.filter = shared(filter, name='filter', borrow=True)

        # 活性化関数
        if activation is None:
            activation = lambda x: x
        self.activation = activation

        # 更新対象パラメタ
        self.params = [self.filter, ]

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
        W = T.tensordot(self.filter, self.T, axes=(0, 2))
        # 各入力の各プーリング領域における最大値を取得する
        result, update = scan(fn=lambda input, W: T.max(input * W, axis=1),
                              sequences=[inputs_symbol],
                              non_sequences=W)
        return result
