# coding: utf-8

import numpy as np
from theano import config, tensor as T
from __hidden import HiddenLayer
from __output import OutputLayerInterface


class SoftMaxLayer(HiddenLayer, OutputLayerInterface):
    def __init__(self, layer_id, n_in, n_out, weight=None, b=None,
                 dtype=config.floatX, is_dropout=False, dropout_rate=0.5):
        if weight is None:
            weight = np.zeros((n_in, n_out), dtype=dtype)
        if b is None:
            b = np.zeros((n_out,), dtype=dtype)

        activation = T.nnet.softmax

        super(SoftMaxLayer, self).__init__(layer_id, n_in, n_out, activation,
                                           weight=weight, b=b, dtype=dtype,
                                           is_dropout=is_dropout,
                                           dropout_rate=dropout_rate)

    def output_argmax(self, input, is_train):
        return T.argmax(self.output(input, is_train), axis=1)

    def cost(self, predict, y):
        return -T.mean(T.log(predict)[T.arange(y.shape[0]), y])

    def error(self, predict, y):
        if predict.ndim != y.ndim or not y.dtype.startswith('int'):
            raise TypeError

        return T.mean(T.neq(predict, y))
