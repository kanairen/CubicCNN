#!/usr/bin/env python
# coding: utf-8

import numpy as np
from __base import BaseLayer


class BaseGridLayer(BaseLayer):
    def __init__(self, layer_id, input_size, c_in, c_out, k, s, p, activation,
                 is_dropout, dropout_rate, cover_all, dim):

        self.input_size = input_size

        self.c_in = c_in
        self.c_out = c_out

        self.k = self._toset(k, dim)
        self.s = self._toset(s, dim)
        self.p = self._toset(p, dim)

        self.output_size = [
            _filter_outsize(size, _k, _s, _p, cover_all=cover_all)
            for size, _k, _s, _p in zip(input_size, self.k, self.s, self.p)]

        n_in = c_in * np.prod(input_size)
        n_out = c_out * np.prod(self.output_size)

        super(BaseGridLayer, self).__init__(layer_id, n_in, n_out,
                                            activation, is_dropout,
                                            dropout_rate)

    def output(self, input, is_train):
        if input.ndim != len(self.input_size) + 2:
            input = input.reshape([input.shape[0], self.c_in] + self.input_size)
        return input

    @staticmethod
    def _toset(x, dim):
        if hasattr(x, '__getitem__'):
            if len(x) == dim:
                return x
            else:
                raise ValueError
        return [x] * dim


class GridLayer2d(BaseGridLayer):
    def __init__(self, layer_id, image_size, c_in, c_out, k, s, p, activation,
                 is_dropout, dropout_rate, cover_all):
        super(GridLayer2d, self).__init__(layer_id, image_size, c_in, c_out, k,
                                          s, p, activation, is_dropout,
                                          dropout_rate, cover_all, dim=2)



def _filter_outsize(size, k, s, p, cover_all=False):
    if cover_all:
        return (size + p * 2 - k + s - 1) // s + 1
    else:
        return (size + p * 2 - k) // s + 1
