#!/usr/bin/env python
# coding: utf-8

import numpy as np
from theano import config, shared, tensor as T
from __base import BaseLayer


class HiddenLayer(BaseLayer):
    def __init__(self, layer_id, n_in, n_out, activation, weight=None,
                 b=None, dtype=config.floatX, is_dropout=False,
                 dropout_rate=0.5):
        super(HiddenLayer, self).__init__(layer_id, n_in, n_out, activation,
                                          is_dropout, dropout_rate)

        if weight is None:
            weight = np.asarray(
                self.rnd.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                 high=np.sqrt(6. / (n_in + n_out)),
                                 size=(n_in, n_out)), dtype=dtype)
            if activation == T.nnet.sigmoid:
                weight *= 4.
        self.W = shared(weight, name='weight{}'.format(layer_id), borrow=True)

        if b is None:
            b = np.zeros(shape=(n_out,), dtype=dtype)
        self.b = shared(b, name='b{}'.format(layer_id), borrow=True)

        self.params = self.W, self.b

    def output(self, input, is_train):
        if input.ndim != 3:
            input = input.flatten(2)

        return self._activate(T.dot(input, self.W) + self.b, is_train)

    def __str__(self):
        return super(HiddenLayer, self).__str__()
