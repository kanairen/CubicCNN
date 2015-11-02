# coding:utf-8

import numpy as np
from theano import config, shared, tensor as T
from src.util import sequence
from filterlayer import FilterLayer

__author__ = 'ren'

"""
Paddingはややこしいので一旦削除
"""


class ConvLayer2d(FilterLayer):
    def __init__(self, img_size, in_channel, out_channel, k_size, stride=1,
                 W=None, T=None, b=None, no_bias=False, filter=None,
                 dtype=config.floatX, activation=None):

        super(ConvLayer2d, self).__init__()

        # 画像サイズ
        img_w, img_h = sequence.pair(img_size)

        # フィルタサイズ
        kw, kh = sequence.pair(k_size)

        # ストライド
        sw, sh = sequence.pair(stride)

        # 入力・出力ユニット数
        n_in = img_w * img_h * in_channel
        n_out = img_w * img_h * out_channel / (stride ** 2)

        # 重み・フィルタ変換行列
        if T is None:
            T = self.init_T(img_w, img_h, kw, kh, sw, sh, n_in, n_out,
                            in_channel, out_channel, dtype=dtype)
        self.T = shared(T, name='T', borrow=True)

        # フィルタベクトル
        if filter is None:
            filter = np.asarray(
                self.rnd.uniform(low=-np.sqrt(1. / in_channel * kw * kh),
                                 high=np.sqrt(1. / in_channel * kw * kh),
                                 size=(out_channel * in_channel * kh * kw)),
                dtype=dtype)
        self.filter = shared(filter, name='filter', borrow=True)

        # バイアスベクトル
        if not no_bias:
            if b is None:
                b = np.zeros(shape=(n_out,), dtype=dtype)
            self.b = shared(b, name='b', borrow=True)

        # 活性化関数
        if activation is None:
            activation = lambda x: x
        self.activation = activation

        # 更新対象パラメタ
        self.params = [self.filter, ]
        if not no_bias:
            self.params.append(self.b)

        # バイアスの有無
        self.no_bias = no_bias

    def update(self, cost, learning_rate=0.01):
        grads = T.grad(cost, self.params)
        return [(p, p - learning_rate * g) for p, g in zip(self.params, grads)]

    def output(self, inputs_symbol):
        # 重み共有のため、フィルタの重みを拝借
        W = T.tensordot(self.filter, self.T, axes=(0, 2))
        return self.activation(T.dot(inputs_symbol, W.T) + self.b)
