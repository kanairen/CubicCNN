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
            h = np.ones((in_channel, in_channel, kh, kw), dtype=dtype)
        self.h = shared(h, name='h', borrow=True)

        super(PoolLayer, self).__init__(img_size, in_channel, in_channel,
                                        k_size, k_size, T, None, True, h,
                                        dtype, activation)

    def update(self, cost, learning_rate=0.01):
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

        # 一時的に4次元テンソルの形に
        inputs_4d = T.reshape(inputs_symbol, (
            inputs_symbol.shape[0], self.in_channel, self.img_w, self.img_h))

        # 畳み込み対象となる画素のみを抽出したテンソル
        col = self.im2col(inputs_4d)

        # テンソル演算 u=Wx
        u = T.tensordot(col, self.h, ((1, 2, 3), (1, 2, 3)))

        # 各出力を一次元配列に戻す
        reshaped_u = T.reshape(u, (inputs_symbol.shape[0], self.n_out))

        # 最大値インデックス配列
        max_args = T.argmax(reshaped_u, axis=1)

        # ゼロ行列
        zeros = T.zeros_like(reshaped_u)

        # 出力の各最大値をゼロ行列の同じインデックスに格納した結果
        z = T.set_subtensor(zeros[T.arange(reshaped_u.shape[0]), max_args],
                            reshaped_u[T.arange(reshaped_u.shape[0]), max_args])

        return z
