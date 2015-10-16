# coding:utf-8

import numpy as np
from theano import tensor as T, shared, config, function

__author__ = 'ren'

rnd = np.random.RandomState(1111)


class Layer(object):
    def __init__(self, n_in, n_out, W=None, b=None, dtype=config.floatX,
                 activation=None):

        # 重み行列
        if W is None:
            W = shared(np.asarray(
                rnd.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                            high=np.sqrt(6. / (n_in + n_out)),
                            size=(n_in, n_out)),
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
            activation = self.relu
        self.activation = activation

        # 入力シンボル
        self.inputs = None

        self.params = self.W, self.b

    @staticmethod
    def relu(x):
        return x * (x > 0)

    def output(self):
        return self.activation(T.dot(self.inputs, self.W) + self.b)
