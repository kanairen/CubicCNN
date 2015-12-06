# coding:utf-8

import numpy as np
import PIL.Image
from theano import config, tensor as T
from filterlayer import FilterLayer, CubicLayer
from src.util.image import gradation_8bit

__author__ = 'ren'


class ConvLayer2d(FilterLayer):
    def __init__(self, img_size, in_channel, out_channel, k_size, stride=1,
                 b=None, no_bias=False, W=None, dtype=config.floatX,
                 activation=None, cover_all=False, is_dropout=False):
        """
        note:画像サイズに対してフィルタサイズが大きいと、後ろの層でエラーが起こる
        """
        super(ConvLayer2d, self).__init__(img_size, in_channel, out_channel,
                                          k_size, stride, b, no_bias, W, dtype,
                                          activation, cover_all, is_dropout)

    def update(self, cost, learning_rate=0.1):
        grads = T.grad(cost, self.params)
        return [(p, p - learning_rate * g) for p, g in zip(self.params, grads)]

    def output(self, inputs_symbol):
        # 一時的に4次元テンソルの形に
        inputs_4d = T.reshape(inputs_symbol, (
            inputs_symbol.shape[0], self.in_channel, self.img_h, self.img_w))

        # 畳み込み対象となる画素のみを抽出したテンソル
        col = self.im2col(inputs_4d)

        # TODO self.bの位置?
        # テンソル演算 u=Wx
        u = np.rollaxis(
            T.tensordot(col, self.W, ((1, 2, 3), (1, 2, 3))) + self.b, 3, 1)

        # 各出力を一次元配列に戻す
        reshaped_u = T.flatten(u, outdim=2)

        # 活性化関数
        z = self.activation(reshaped_u)

        if self.is_dropout:
            if self.is_train:
                z *= self.srnd.binomial(size=z.shape, p=0.5)
            else:
                z *= self.dropout_rate

        return z

    def filter_image(self):
        filters = gradation_8bit(self.W.get_value().copy())
        f_images = []
        for out_c in xrange(self.out_channel):
            for in_c in xrange(self.in_channel):
                f = filters[out_c][in_c]
                f_img = PIL.Image.fromarray(f)
                f_images.append(f_img)
        return f_images


class ConvLayer3d(CubicLayer):
    def __init__(self, box_size, in_channel, out_channel, k_size, stride=1,
                 b=None, no_bias=False, W=None, dtype=config.floatX,
                 activation=None, cover_all=False, is_dropout=False):
        """
        note:画像サイズに対してフィルタサイズが大きいと、後ろの層でエラーが起こる
        """
        super(ConvLayer3d, self).__init__(box_size, in_channel, out_channel,
                                          k_size, stride, b, no_bias, W, dtype,
                                          activation, cover_all, is_dropout)

    def update(self, cost, learning_rate=0.1):
        grads = T.grad(cost, self.params)
        return [(p, p - learning_rate * g) for p, g in zip(self.params, grads)]

    def output(self, inputs_symbol):
        # 一時的に4次元テンソルの形に
        inputs_5d = T.reshape(inputs_symbol, (
            inputs_symbol.shape[0], self.in_channel, self.box_z, self.box_y,
            self.box_x))

        # 畳み込み対象となる画素のみを抽出したテンソル
        col = self.box2col(inputs_5d)

        # TODO +self.bの位置?
        # テンソル演算 u=Wx
        u = np.rollaxis(
            T.tensordot(col, self.W, ((1, 2, 3, 4), (1, 2, 3, 4))) + self.b, 4,
            1)

        # 各出力を一次元配列に戻す
        reshaped_u = T.flatten(u, outdim=2)

        # 活性化関数
        z = self.activation(reshaped_u)

        if self.is_dropout:
            if self.is_train:
                z *= self.srnd.binomial(size=z.shape, p=0.5)
            else:
                z *= self.dropout_rate

        return z
