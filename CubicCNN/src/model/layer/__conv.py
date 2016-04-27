#!/usr/bin/env python
# coding: utf-8

import numpy as np
from theano import config, shared
from theano.tensor.nnet.conv import conv2d
from theano.tensor.nnet.Conv3D import conv3D
from __grid import GridLayer2d, GridLayer3d


class ConvLayer2d(GridLayer2d):
    def __init__(self, layer_id, image_size, activation, c_in, c_out, k,
                 s=(1, 1), p=(0, 0), filters=None, b=None, border_mode='valid',
                 dtype=config.floatX, is_dropout=False, dropout_rate=0.5):

        super(ConvLayer2d, self).__init__(layer_id, image_size, c_in, c_out, k,
                                          s, p, activation, is_dropout,
                                          dropout_rate, cover_all=False)

        if filters is None:
            kw, kh = self.k
            f_in = c_in * kh * kw
            f_out = c_out * kh * kw
            w_bound = np.sqrt(6. / (f_in + f_out))
            filters = np.asarray(self.rnd.uniform(low=-w_bound,
                                                  high=w_bound,
                                                  size=(c_out, c_in, kh, kw)),
                                 dtype=dtype)
        self.filters = shared(filters, name='filters{}'.format(layer_id),
                              borrow=True)

        if b is None:
            b = np.zeros((c_out,), dtype=dtype)
        self.b = shared(b, name='b{}'.format(layer_id), borrow=True)

        self.params = self.filters, self.b

        self.border_mode = border_mode

    def output(self, input, is_train):
        input = super(ConvLayer2d, self).output(input, is_train)

        u = conv2d(input, self.filters,
                   filter_shape=(self.c_out, self.c_in, self.k[1], self.k[0]),
                   border_mode=self.border_mode,
                   subsample=self.s)
        # TODO バイアスの挿入位置がここでほんとうに正しいのかテスト
        return self._activate(u + self.b.dimshuffle('x', 0, 'x', 'x'), is_train)

    def __str__(self):
        return super(ConvLayer2d, self).__str__()


class ConvLayer3d(GridLayer3d):
    def __init__(self, layer_id, shape_size, activation, c_in, c_out, k,
                 s=(1, 1, 1), p=(0, 0, 0), filters=None, b=None,
                 border_mode='valid', dtype=config.floatX, is_dropout=False,
                 dropout_rate=0.5):

        super(ConvLayer3d, self).__init__(layer_id, shape_size, c_in, c_out, k,
                                          s, p, activation, is_dropout,
                                          dropout_rate, cover_all=False)

        if filters is None:
            f_in = c_in * np.prod(self.k)
            f_out = c_out * np.prod(self.k)
            w_bound = np.sqrt(6. / (f_in + f_out))
            # shape = (c_out, self.k[2], c_in, self.k[1], self.k[0])
            shape = (c_out, self.k[2], self.k[1], self.k[0], c_in)
            filters = np.asarray(self.rnd.uniform(low=-w_bound,
                                                  high=w_bound,
                                                  size=shape),
                                 dtype=dtype)
        self.filters = shared(filters, name='filters{}'.format(layer_id),
                              borrow=True)

        if b is None:
            b = np.zeros((c_out,), dtype=dtype)
        self.b = shared(b, name='b{}'.format(layer_id), borrow=True)

        self.params = self.filters, self.b

        self.border_mode = border_mode

    def output(self, input, is_train):
        input = super(ConvLayer3d, self).output(input, is_train)

        u = conv3D(input.dimshuffle(0, 2, 3, 4, 1), self.filters,
                   self.b, d=self.s).dimshuffle(0, 4, 1, 2, 3)

        # TODO バイアスの挿入位置がここでほんとうに正しいのかテスト
        return self._activate(u, is_train)

    def __str__(self):
        return super(ConvLayer3d, self).__str__()
