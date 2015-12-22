# coding:utf-8

import numpy as np
import six
from theano import config, shared, tensor as T
from src.model.layer.baselayer import BaseLayer
from src.util.sequence import pair, trio

__author__ = 'ren'


class FilterLayer(BaseLayer):
    def __init__(self, img_size, in_channel, out_channel, k_size, stride=1,
                 b=None, no_bias=False, W=None, dtype=config.floatX,
                 activation=None, cover_all=False, is_dropout=False,
                 dropout_rate=0.5):

        super(FilterLayer, self).__init__(is_dropout, dropout_rate)

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
        out_h = conv_outsize(img_h, kh, sh, ph, cover_all)
        out_w = conv_outsize(img_w, kw, sw, pw, cover_all)
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

    def __str__(self):
        return super(FilterLayer, self).__str__() + \
               " n_in : {:<5}".format(self.n_in) + \
               " n_out : {:<5}".format(self.n_out) + \
               " in_channel : {:<5}".format(self.in_channel) + \
               " out_channel : {:<5}".format(self.out_channel) + \
               " kernel : {:<5}".format((self.kw, self.kh)) + \
               " stride : {:<5}".format((self.sw, self.sh)) + \
               " pad : {:<5}".format((self.pw, self.ph)) + \
               " out_img_size : {:<5}".format(self.output_img_size())


class CubicLayer(BaseLayer):
    def __init__(self, box_size, in_channel, out_channel, k_size, stride=1,
                 b=None, no_bias=False, W=None, dtype=config.floatX,
                 activation=None, cover_all=False, is_dropout=False,
                 dropout_rate=0.5):

        super(CubicLayer, self).__init__(is_dropout, dropout_rate)

        # 画像サイズ
        box_x, box_y, box_z = trio(box_size)
        self.box_x = box_x
        self.box_y = box_y
        self.box_z = box_z

        # チャネル数
        self.in_channel = in_channel
        self.out_channel = out_channel

        # フィルタサイズ
        kx, ky, kz = trio(k_size)
        self.kx = kx
        self.ky = ky
        self.kz = kz

        # ストライド
        sx, sy, sz = trio(stride)
        self.sx = sx
        self.sy = sy
        self.sz = sz

        # パディング
        px, py, pz = trio(0)
        self.px = px
        self.py = py
        self.pz = pz

        # 出力画像サイズ
        out_x = conv_outsize(box_x, kx, sx, px, cover_all)
        out_y = conv_outsize(box_y, ky, sy, py, cover_all)
        out_z = conv_outsize(box_z, kz, sz, pz, cover_all)
        self.out_x = out_x
        self.out_y = out_y
        self.out_z = out_z

        # 入力・出力ユニット数
        n_in = box_x * box_y * box_z * in_channel
        n_out = out_x * out_y * out_z * out_channel
        self.n_in = n_in
        self.n_out = n_out

        # フィルタベクトル
        if W is None:
            W = np.asarray(
                    self.rnd.uniform(
                            low=-np.sqrt(1. / in_channel * kx * ky * kz),
                            high=np.sqrt(1. / in_channel * kx * ky * kz),
                            size=(out_channel, in_channel, kx, ky, kz)),
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
        super(CubicLayer, self).update(cost, learning_rate)

    def output(self, inputs_symbol):
        super(CubicLayer, self).output(inputs_symbol)

    def output_box_size(self):
        return self.out_x, self.out_y, self.out_z

    def box2col(self, box, pval=0):
        n, c, z, y, x = box.shape

        col = T.zeros((n, c, self.kz, self.ky, self.kx, self.out_z, self.out_y,
                       self.out_x), dtype=box.dtype)

        for i in six.moves.range(self.kz):
            i_lim = i + self.sz * self.out_z
            for j in six.moves.range(self.ky):
                j_lim = j + self.sy * self.out_y
                for k in six.moves.range(self.kx):
                    k_lim = k + self.sx * self.out_x
                    col = T.set_subtensor(col[:, :, i, j, k, :, :, :],
                                          box[:, :, i:i_lim:self.sz,
                                          j:j_lim:self.sy, k:k_lim:self.sx])

        return col

    def __str__(self):
        return super(CubicLayer, self).__str__() + \
               " n_in : {:<8}".format(self.n_in) + \
               " n_out : {:<8}".format(self.n_out) + \
               " in_channel : {:<5}".format(self.in_channel) + \
               " out_channel : {:<5}".format(self.out_channel) + \
               " kernel : {:<8}".format((self.kx, self.ky, self.kz)) + \
               " stride : {:<8}".format((self.sx, self.sy, self.sz)) + \
               " pad : {:<8}".format((self.px, self.py, self.pz)) + \
               " out_img_size : {:<8}".format(self.output_box_size())


def conv_outsize(size, k, s, p, cover_all=False):
    if cover_all:
        return (size + p * 2 - k + s - 1) // s + 1
    else:
        return (size + p * 2 - k) // s + 1
