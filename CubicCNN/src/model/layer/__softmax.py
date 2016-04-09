#!/usr/bin/env python
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

    def output(self, input, is_train):
        return super(SoftMaxLayer, self).output(input, is_train)
