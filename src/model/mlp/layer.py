# coding:utf-8

import numpy as np
from theano import tensor as T, shared, config, function

__author__ = 'ren'

rnd = np.random.RandomState(1111)


class Layer(object):
    def __init__(self, n_in, n_out, W=None, b=None, activation=None):
        if W is None:
            W = shared(np.asarray(
                rnd.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                            high=np.sqrt(6. / (n_in + n_out)),
                            size=(n_in, n_out)),
                dtype=config.float32
            ), name='W', borrow=True)

        if b is None:
            b = shared(np.zeros(shape=(n_out,), dtype=config.float32),
                       name='b', borrow=True)

        if activation is None:
            activation = self.relu

        self.W = W
        self.b = b
        self.params = self.W, self.b

        self.inputs = T.fvector
        self.outputs = activation(np.dot(self.inputs, self.W) + self.b)

        self.softmax = T.nnet.softmax(T.dot(self.inputs, self.W) + self.b)
        self.max_arg = T.argmax(self.softmax, axis=1)

    def forward(self, inputs, updates=(), givens={}):
        return function(inputs=[self.inputs],
                        outputs=self.outputs,
                        givens=givens,
                        updates=updates)(inputs)

    @staticmethod
    def relu(x):
        return x * (x > 0)

    @staticmethod
    def accuracy(pred, answer):
        assert len(pred) == len(answer)
        return function(T.mean(T.eq(pred, answer)))
