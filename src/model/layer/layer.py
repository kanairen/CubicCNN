# coding:utf-8

import numpy as np
from theano import tensor as T, shared, config
from src.util.activation import relu
from src.interface.layerinterface import LayerInterface

__author__ = 'ren'

rnd = np.random.RandomState(1111)


class Layer(LayerInterface):
    def __init__(self, n_in, n_out, W=None, b=None, dtype=config.floatX,
                 activation=None):

        # 重み行列
        if W is None:
            W = shared(np.asarray(
                rnd.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                            high=np.sqrt(6. / (n_in + n_out)),
                            size=(n_out, n_in)),
                dtype=dtype
            ), name='W', borrow=True)

            if activation == T.nnet.sigmoid:
                W *= 4.
        self.W = W

        # バイアスベクトル
        if b is None:
            b = shared(np.zeros(shape=(n_out,), dtype=dtype), name='b',
                       borrow=True)
        self.b = b

        # 活性化関数
        if activation is None:
            activation = relu
        self.activation = activation

        self.params = self.W, self.b

    def update(self, cost, learning_rate=0.01):
        grads = T.grad(cost, self.params)
        return [(p, p - learning_rate * g) for p, g in zip(self.params, grads)]

    def output(self, inputs_symbol):
        return self.activation(T.dot(inputs_symbol, self.W.T) + self.b)
