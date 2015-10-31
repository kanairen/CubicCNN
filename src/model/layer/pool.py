# coding:utf-8

import numpy as np
from theano import config, tensor as T,scan

from src.model.layer.conv import ConvLayer2d
from src.util.activation import relu
from src.util.sequence import pair

__author__ = 'ren'


class PoolLayer(ConvLayer2d):
    POOL_MAX = 0
    POOL_AVERAGE = 1

    def __init__(self, img_size, in_channel, k_size, pad=0, dtype=config.floatX,
                 activation=relu, pool_type=POOL_MAX):

        kw, kh = pair(k_size)

        # フィルタベクトル
        filter = np.ones((in_channel * in_channel * kh * kw), dtype=dtype)

        super(PoolLayer, self).__init__(img_size, in_channel, in_channel,
                                        k_size, k_size, pad, None, None, filter,
                                        dtype, activation)

        self.pool_type = pool_type

    def update(self, cost, learning_rate=0.01):
        return None

    def output(self, inputs_symbol):
        self.filtering()

        linear = T.dot(inputs_symbol, self.W.T) + self.b
        if self.pool_type == PoolLayer.POOL_MAX:

            def max_pool_output(v):
                max_arg = T.argmax(v)
                zeros = T.zeros_like(v)
                return T.set_subtensor(zeros[max_arg],v[max_arg])

            result,update = scan(fn=max_pool_output,sequences=linear)

            return result
        elif self.pool_type == PoolLayer.POOL_AVERAGE:
            return T.mean(linear, axis=1,keepdims=True)
        else:
            raise RuntimeError("pool_type is invalid.")
