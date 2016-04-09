#!/usr/bin/env python
# coding: utf-8

from theano.tensor.signal.downsample import max_pool_2d
from __grid import GridLayer2d


class MaxPoolLayer2d(GridLayer2d):
    def __init__(self, layer_id, image_size, activation, c_in, k,
                 s=None, p=(0, 0), ignore_border=False, mode='max',
                 is_dropout=False, dropout_rate=0.5):
        if s is None:
            s = k

        super(MaxPoolLayer2d, self).__init__(layer_id, image_size, c_in, c_in,
                                             k, s, p, activation, is_dropout,
                                             dropout_rate, cover_all=True)

        self.params = []
        self.ignore_border = ignore_border
        self.mode = mode

    def output(self, input, is_train):
        input = super(MaxPoolLayer2d, self).output(input, is_train)

        u = max_pool_2d(input, ds=self.k,
                        ignore_border=self.ignore_border,
                        st=self.s,
                        padding=self.p,
                        mode=self.mode)
        return self._activate(u, is_train)

    def __str__(self):
        return super(MaxPoolLayer2d, self).__str__()
