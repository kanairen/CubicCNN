# coding:utf-8

import numpy as np
from theano import tensor as T, shared, config, function

__author__ = 'ren'

rnd = np.random.RandomState(1111)


class Layer(object):
    def __init__(self, n_in, n_out, inputs, W=None, b=None, activation=None):
        if W is None:
            W = shared(np.asarray(
                rnd.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                            high=np.sqrt(6. / (n_in + n_out)),
                            size=(n_in, n_out)),
                dtype=config.floatX
            ), name='W', borrow=True)

            if activation == T.nnet.sigmoid:
                W *= 4.

        if b is None:
            b = shared(np.zeros(shape=(n_out,), dtype=config.floatX),
                       name='b', borrow=True)

        if activation is None:
            activation = self.relu

        if inputs is None:
            inputs = T.fmatrix('inputs')

        self.W = W
        self.b = b
        self.params = self.W, self.b

        self.inputs = inputs
        self.outputs = activation(T.dot(self.inputs, self.W) + self.b)

        self.softmax = T.nnet.softmax(self.outputs)
        self.max_arg = T.argmax(self.softmax, axis=1)

    def forward(self, inputs, updates=(), givens={}):
        print "forward"
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
