# coding:utf-8

import numpy as np
from theano import config, shared, tensor as T
from layer import rnd
from src.util import sequence
import six

__author__ = 'ren'


# TODO ConvLayerTest 30min(12:00)
# TODO プーリング層


class ConvLayer2d(object):
    def __init__(self, img_size, in_channel, out_channel, k_size, stride=1,
                 pad=0, W=None, b=None, no_bias=False, filter=None,
                 dtype=config.floatX, activation=None):

        # 画像サイズ
        self.img_w, self.img_h = sequence.pair(img_size)

        # フィルタサイズ
        self.kw, self.kh = sequence.pair(k_size)

        # ストライド
        self.sw, self.sh = sequence.pair(stride)

        # パディング
        self.pw, self.ph = sequence.pair(pad)

        # 入力・出力チャネル
        self.in_channel = in_channel
        self.out_channel = out_channel

        # 入力・出力ユニット数
        self.n_in = self.img_w * self.img_h * in_channel
        self.n_out = self.img_w * self.img_h * out_channel / (stride ** 2)

        if W is None:
            W = np.zeros(shape=(self.n_out, self.n_in), dtype=dtype)
        self.W = shared(W, name='W', borrow=True)

        # フィルタベクトル
        if filter is None:
            filter = np.asarray(
                rnd.uniform(low=-np.sqrt(1. / in_channel * self.kw * self.kh),
                            high=np.sqrt(1. / in_channel * self.kw * self.kh),
                            size=(
                                out_channel * in_channel * self.kh * self.kw)),
                dtype=dtype)
        self.filter = shared(filter, name='filter', borrow=True)

        # バイアスベクトル
        if not no_bias:
            if b is None:
                b = shared(np.zeros(shape=(self.n_out,), dtype=dtype), name='b',
                           borrow=True)
            self.b = b

        # 活性化関数
        if activation is None:
            activation = lambda x: x
        self.activation = activation

        # 更新対象パラメタ
        self.params = [self.W, ]
        if not no_bias:
            self.params.append(self.b)

        # バイアスの有無
        self.no_bias = no_bias

    def update(self, cost, learning_rate=0.01):
        grads = T.grad(cost, self.params)
        return [(p, p - learning_rate * g) for p, g in zip(self.params, grads)]

    def output(self, inputs_symbol):
        # 重み共有のため、毎回フィルタの重みを拝借
        self.filtering()
        return self.activation(T.dot(inputs_symbol, self.W.T) + self.b)

    def filtering(self):
        h_outsize = self.conv_outsize(self.img_h, self.kh, self.sh, self.ph,
                                      True)
        w_outsize = self.conv_outsize(self.img_w, self.kw, self.sw, self.pw,
                                      True)
        W = self.W.get_value()
        filter = self.filter.get_value()

        for k in six.moves.range(self.in_channel):
            for j in six.moves.range(0, h_outsize, self.sh):
                for i in six.moves.range(0, w_outsize, self.sw):
                    for m in six.moves.range(self.out_channel):
                        for kh in six.moves.range(self.kh):
                            for kw in six.moves.range(self.kw):
                                W[(
                                      j * (w_outsize - self.kw) + i) + (
                                      m * self.kw * self.kh) + (
                                      kh * self.kw + kw)][
                                    k * j * i] = filter[m * (
                                    self.in_channel * self.kh * self.kw) + k * (
                                                            self.kh * self.kw) + kh * self.kw + kw]
        self.W.set_value(W)

    @staticmethod
    def conv_outsize(size, k, s, p, cover_all=False):
        if cover_all:
            return (size + p * 2 - k + s - 1) // s + 1
        else:
            return (size + p * 2 - k) // s + 1
