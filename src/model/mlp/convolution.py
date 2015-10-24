# coding:utf-8

import numpy as np
from theano import config, shared, tensor as T, function, pp
from layer import Layer, rnd
from src.util import conv
import six

__author__ = 'ren'


# TODO ConvLayerTest 30min(12:00)
# TODO プーリング層


class ConvLayer2d(object):
    def __init__(self, img_size, k_size, in_channel, out_channel, stride=1,
                 pad=0, W=None, b=None, dtype=config.floatX, activation=None):

        # 画像サイズ
        self.img_w, self.img_h = conv.pair(img_size)

        # フィルタサイズ
        self.kw, self.kh = conv.pair(k_size)

        # ストライド
        self.sw, self.sh = conv.pair(stride)

        # パディング
        self.pw, self.ph = conv.pair(pad)

        # 入力・出力チャネル
        self.in_channel = in_channel
        self.out_channel = out_channel

        # 入力・出力ユニット数
        self.n_in = self.img_w * self.img_h * in_channel
        self.n_out = self.img_w * self.img_h * out_channel

        if W is None:
            W = np.zeros(shape=(self.n_out, self.n_in), dtype=dtype)
        self.W = shared(W, name='W', borrow=True)
        print self.W.name, self.W.get_value().shape

        # フィルタベクトル
        self.filter = shared(np.asarray(
            rnd.uniform(low=-np.sqrt(1. / in_channel * self.kw * self.kh),
                        high=np.sqrt(1. / in_channel * self.kw * self.kh),
                        size=(out_channel * in_channel * self.kh * self.kw)),
            dtype=dtype
        ), name='filter', borrow=True)
        print self.filter.name, self.filter.get_value().shape

        # バイアスベクトル
        if b is None:
            b = shared(np.zeros(shape=(self.n_out,), dtype=dtype), name='b',
                       borrow=True)
        self.b = b

        if activation is None:
            activation = lambda x: x
        self.activation = activation

        self.params = self.W, self.b

    def update(self, cost, learning_rate=0.01):
        grads = T.grad(cost, self.params)
        return [(p, p - learning_rate * g) for p, g in zip(self.params, grads)]

    def output(self, inputs_symbol):
        # 重み共有のため、毎回フィルタの重みを拝借
        self.conv_filter()
        return self.activation(T.dot(inputs_symbol, self.W.T) + self.b)

    def conv_filter(self):
        h_outsize = conv.conv_outsize(self.img_h, self.kh, self.sh, self.ph,
                                      True)
        w_outsize = conv.conv_outsize(self.img_w, self.kw, self.sw, self.pw,
                                      True)
        filter = self.filter.get_value()
        W = self.W.get_value()
        for m in six.moves.range(self.out_channel):
            for j in six.moves.range(h_outsize):
                for i in six.moves.range(w_outsize):
                    for k in six.moves.range(self.in_channel):
                        for kh in six.moves.range(self.kh):
                            for kw in six.moves.range(self.kw):
                                W[m * j * i][
                                    k * ((j + kh * self.sh) * self.img_w + (
                                        i + kw * self.sw))] = filter[
                                    m * k * kh * kw]
        self.W.set_value(W)
