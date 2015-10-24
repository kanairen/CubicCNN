# coding:utf-8

import numpy as np
from theano import config, shared, tensor as T, function, pp
from layer import Layer, rnd
from src.util import conv

__author__ = 'ren'


# TODO ConvLayerTest 30min(12:00)
# TODO プーリング層



class ConvLayer2d(object):
    def __init__(self, img_size, k_size, in_channel, out_channel, stride=1,
                 W=None, b=None, dtype=config.floatX, activation=None):
        # 画像サイズ
        w, h = conv.pair(img_size)

        # 入力・出力ユニット数
        self.n_in = w * h * in_channel
        self.n_out = w * h * out_channel

        # フィルタサイズ
        kw, kh = conv.pair(k_size)

        # フィルタベクトル
        self.h = shared(np.asarray(
            rnd.uniform(low=-np.sqrt(1. / in_channel * kw * kh),
                        high=np.sqrt(1. / in_channel * kw * kh),
                        size=(out_channel * in_channel * kh * kw)),
            dtype=dtype
        ), name='h', borrow=True)

        # 結合行列
        # 値としてスパースなベクトルではなく。ｈのインデックスを保持
        t = np.zeros(
            (self.n_in, self.n_out, out_channel * in_channel * kh * kw),
            dtype=dtype)

        # ユニットの結合を定義
        for i in range(w - kw):
            for j in range(h - kh):
                for c in range(in_channel * out_channel):
                    for k_h in range(kh):
                        for k_w in range(kw):
                            t[i + k_w * stride][j + k_h * stride][
                                (kw * k_h + k_w) * c] = 1
        self.t = shared(t, name='t', borrow=True)

        # バイアスベクトル
        if b is None:
            b = shared(np.zeros(shape=(self.n_out,), dtype=dtype), name='b',
                       borrow=True)
        self.b = b

        if activation is None:
            activation = lambda x: x
        self.activation = activation

        self.params = self.h, self.b

    def update(self, cost, learning_rate=0.01):
        # TODO もしかしたら、勾配の修正方法が正しくないかも
        # TODO 勾配の更新がなされていない
        grads = T.grad(cost, self.params)
        return [(p, p - learning_rate * g) for p, g in zip(self.params, grads)]

    def output(self, inputs_symbol):
        # 重み共有のため、毎回フィルタの重みを拝借
        return self.activation(
            T.dot(inputs_symbol, T.sum(self.t * self.h, axis=2) + self.b))

