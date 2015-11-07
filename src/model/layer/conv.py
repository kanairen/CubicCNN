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
        inputs_4d = T.reshape(inputs_symbol, (
            inputs_symbol.shape[0], self.in_channel, self.img_w, self.img_h))

        col = self.im2col(inputs_4d)

        u = T.tensordot(col, self.h, ((1, 2, 3), (1, 2, 3))) + self.b

        z = self.activation(u)

        return T.reshape(z, (inputs_symbol.shape[0], self.n_out))
