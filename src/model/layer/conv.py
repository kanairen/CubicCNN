# coding:utf-8

from theano import config, tensor as T
from filterlayer import FilterLayer

__author__ = 'ren'


class ConvLayer2d(FilterLayer):
    def __init__(self, img_size, in_channel, out_channel, k_size, stride=1,
                 T=None, b=None, no_bias=False, h=None,
                 dtype=config.floatX, activation=None):
        super(ConvLayer2d, self).__init__(img_size, in_channel, out_channel,
                                          k_size, stride, T, b, no_bias, h,
                                          dtype, activation)

    def update(self, cost, learning_rate=0.01):
        grads = T.grad(cost, self.params)
        return [(p, p - learning_rate * g) for p, g in zip(self.params, grads)]

    def output(self, inputs_symbol):
        # 一時的に4次元テンソルの形に
        inputs_4d = T.reshape(inputs_symbol, (
            inputs_symbol.shape[0], self.in_channel, self.img_w, self.img_h))

        # 畳み込み対象となる画素のみを抽出したテンソル
        col = self.im2col(inputs_4d)

        # テンソル演算 u=Wx
        u = T.tensordot(col, self.h, ((1, 2, 3), (1, 2, 3))) + self.b

        # 各出力を一次元配列に戻す
        reshaped_u = T.reshape(u, (inputs_symbol.shape[0], self.n_out))

        # 活性化関数
        z = self.activation(reshaped_u)

        return z
