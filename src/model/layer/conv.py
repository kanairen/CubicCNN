# coding:utf-8

from theano import config, tensor as T
from filterlayer import FilterLayer

__author__ = 'ren'

"""
Paddingはややこしいので一旦削除
"""


class ConvLayer2d(FilterLayer):
    def __init__(self, img_size, in_channel, out_channel, k_size, stride=1,
                 T=None, b=None, no_bias=False, filter=None,
                 dtype=config.floatX, activation=None):
        super(ConvLayer2d, self).__init__(img_size, in_channel, out_channel,
                                          k_size, stride, T, b, no_bias, filter,
                                          dtype, activation)

    def update(self, cost, learning_rate=0.01):
        grads = T.grad(cost, self.params)
        return [(p, p - learning_rate * g) for p, g in zip(self.params, grads)]

    def output(self, inputs_symbol):
        # 重み共有のため、フィルタの重みを拝借
        W = T.tensordot(self.filter, self.T, axes=(0, 2))
        return self.activation(T.dot(inputs_symbol, W.T) + self.b)
