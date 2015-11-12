# coding:utf-8

import numpy as np
import six
from theano import config, shared, tensor as T

from src.model.layer.baselayer import BaseLayer
from src.util.sequence import pair

__author__ = 'ren'


class FilterLayer(BaseLayer):
    def __init__(self, img_size, in_channel, out_channel, k_size, stride=1,
                 T=None, b=None, no_bias=False, W=None,
                 dtype=config.floatX, activation=None, cover_all=False,
                 is_dropout=False):

        super(FilterLayer, self).__init__(is_dropout)

        # 画像サイズ
        img_w, img_h = pair(img_size)
        self.img_w = img_w
        self.img_h = img_h

        # チャネル数
        self.in_channel = in_channel
        self.out_channel = out_channel

        # フィルタサイズ
        kw, kh = pair(k_size)
        self.kw = kw
        self.kh = kh

        # ストライド
        sw, sh = pair(stride)
        self.sw = sw
        self.sh = sh

        # パディング
        pw, ph = pair(0)
        self.pw = pw
        self.ph = ph

        # 出力画像サイズ
        out_h = self.conv_outsize(img_h, kh, sh, ph, cover_all)
        out_w = self.conv_outsize(img_w, kw, sw, pw, cover_all)
        self.out_h = out_h
        self.out_w = out_w

        # 入力・出力ユニット数
        n_in = img_w * img_h * in_channel
        n_out = out_w * out_h * out_channel
        self.n_in = n_in
        self.n_out = n_out

        # フィルタベクトル
        if W is None:
            W = np.asarray(
                self.rnd.uniform(low=-np.sqrt(1. / in_channel * kw * kh),
                                 high=np.sqrt(1. / in_channel * kw * kh),
                                 size=(out_channel, in_channel, kh, kw)),
                dtype=dtype)
        self.W = shared(W, name='W', borrow=True)

        # バイアスベクトル
        if not no_bias:
            if b is None:
                b = np.zeros(shape=(out_channel,), dtype=dtype)
            self.b = shared(b, name='b', borrow=True)

        # 活性化関数
        if activation is None:
            activation = lambda x: x
        self.activation = activation

        # 更新対象パラメタ
        self.params = [self.W, ]
        if not no_bias:
            self.params.append(self.b)

    def update(self, cost, learning_rate):
        super(FilterLayer, self).update(cost, learning_rate)

    def output(self, inputs_symbol):
        super(FilterLayer, self).output(inputs_symbol)

    def output_img_size(self):
        return self.out_w, self.out_h

    @staticmethod
    def conv_outsize(size, k, s, p, cover_all=False):
        if cover_all:
            return (size + p * 2 - k + s - 1) // s + 1
        else:
            return (size + p * 2 - k) // s + 1

    def im2col(self, img, pval=0):
        n, c, h, w = img.shape

        # img = np.pad(img, ((0, 0), (0, 0), (self.ph, self.ph + self.sh - 1),
        #                    (self.pw, self.pw + self.sw - 1)),
        #              mode='constant', constant_values=(pval,))

        col = T.zeros((n, c, self.kh, self.kw, self.out_h, self.out_w),
                      dtype=img.dtype)

        for i in six.moves.range(self.kh):
            i_lim = i + self.sh * self.out_h
            for j in six.moves.range(self.kw):
                j_lim = j + self.sw * self.out_w
                col = T.set_subtensor(col[:, :, i, j, :, :],
                                      img[:, :, i:i_lim:self.sh,
                                      j:j_lim:self.sw])

        return col
