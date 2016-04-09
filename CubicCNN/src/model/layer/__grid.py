#!/usr/bin/env python
# coding: utf-8

import numpy as np
from __base import BaseLayer


class GridLayer2d(BaseLayer):
    def __init__(self, layer_id, image_size, c_in, c_out, k, s, p, activation,
                 is_dropout, dropout_rate, cover_all):

        self.image_size = image_size

        self.c_in = c_in
        self.c_out = c_out

        self.k = self._pair(k)
        self.s = self._pair(s)
        self.p = self._pair(p)

        img_w, img_h = image_size
        kw, kh = self.k
        sw, sh = self.s
        pw, ph = self.p

        out_w = self._filter_outsize(img_w, kw, sw, pw, cover_all=cover_all)
        out_h = self._filter_outsize(img_h, kh, sh, ph, cover_all=cover_all)

        self.output_image_size = out_w, out_h

        n_in = c_in * np.prod(image_size)
        n_out = c_out * np.prod(self.output_image_size)

        super(GridLayer2d, self).__init__(layer_id, n_in, n_out, activation,
                                          is_dropout, dropout_rate)

    def output(self, input, is_train):
        if input.ndim != 4:
            input = input.reshape(
                (input.shape[0], self.c_in, self.image_size[0],
                 self.image_size[1]))
        return input

    @staticmethod
    def _filter_outsize(size, k, s, p, cover_all=False):
        if cover_all:
            return (size + p * 2 - k + s - 1) // s + 1
        else:
            return (size + p * 2 - k) // s + 1

    @staticmethod
    def _pair(x):
        if hasattr(x, '__getitem__'):
            if len(x) == 2:
                return x
            else:
                raise ValueError
        return x, x
