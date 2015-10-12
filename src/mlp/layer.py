# coding:utf-8

import numpy as np
from theano import tensor as T, shared, config, function

__author__ = 'ren'

rnd = np.random.RandomState(1111)


class Layer(object):
    def __init__(self, n_in, n_out, W=None, b=None, activation=None,
                 updates=None, learning_rate=0.001):
        if W is None:
            W = shared(np.asarray(
                rnd.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                            high=np.sqrt(6. / (n_in + n_out)),
                            size=(n_in, n_out)),
                dtype=config.float32
            ), name='W', borrow=True)

        if b is None:
            b = shared(np.zeros(shape=(n_out,), dtype=config.float32), name='b')

        if activation is None:
            activation = self.relu

        if updates is None:
            W_grad = T.grad()
            updates = [(self.W,self.W-learning_rate*),
                       (self.b,)]

        self.W = W
        self.b = b
        self.activation = activation

        self.s_input = T.fvector

        self.forward = function(inputs=self.s_input,
                                outputs=self.activation(np.dot(self.W, self.s_input) + self.b),
                                updates=updates)

    def forward(self, inputs):

    def backward(self):

    @staticmethod
    def relu(x):
        return x * (x > 0)
