#!/usr/bin/env python
# coding: utf-8

import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from theano.tensor.signal.downsample import DownsampleFactorMax
from __grid import GridLayer2d, GridLayer3d


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

        u = pool_2d(input, ds=self.k,
                    ignore_border=self.ignore_border,
                    st=self.s,
                    padding=self.p,
                    mode=self.mode)
        return self._activate(u, is_train)

    def __str__(self):
        return super(MaxPoolLayer2d, self).__str__()


class MaxPoolLayer3d(GridLayer3d):
    def __init__(self, layer_id, shape_size, activation, c_in, k,
                 s=None, p=(0, 0, 0), ignore_border=False, mode='max',
                 is_dropout=False, dropout_rate=0.5):
        if s is None:
            s = k

        super(MaxPoolLayer3d, self).__init__(layer_id, shape_size, c_in, c_in,
                                             k, s, p, activation, is_dropout,
                                             dropout_rate, cover_all=True)

        self.params = []
        self.ignore_border = ignore_border
        self.mode = mode

    def output(self, input, is_train):
        input = super(MaxPoolLayer3d, self).output(input, is_train)

        u = max_pool_3d(input, ds=self.k, ignore_border=self.ignore_border)

        return self._activate(u, is_train)
    
    def __str__(self):
        return super(MaxPoolLayer3d, self).__str__()


def max_pool_3d(input, ds, ignore_border=False):

    if input.ndim < 3:
        raise NotImplementedError('max_pool_3d requires a dimension >= 3')

    vid_dim = input.ndim

    frame_shape = input.shape[-2:]

    batch_size = T.prod(input.shape[:-2])
    batch_size = T.shape_padright(batch_size, 1)

    new_shape = T.cast(T.join(0, batch_size,
                              T.as_tensor([1, ]),
                              frame_shape), 'int32')
    input_4D = T.reshape(input, new_shape, ndim=4)

    op = DownsampleFactorMax((ds[1], ds[2]), ignore_border)
    output = op(input_4D)
    outshape = T.join(0, input.shape[:-2], output.shape[-2:])
    out = T.reshape(output, outshape, ndim=input.ndim)

    shufl = (
    list(range(vid_dim - 3)) + [vid_dim - 2] + [vid_dim - 1] + [vid_dim - 3])
    input_time = out.dimshuffle(shufl)
    vid_shape = input_time.shape[-2:]

    batch_size = T.prod(input_time.shape[:-2])
    batch_size = T.shape_padright(batch_size, 1)

    new_shape = T.cast(T.join(0, batch_size,
                              T.as_tensor([1, ]),
                              vid_shape), 'int32')
    input_4D_time = T.reshape(input_time, new_shape, ndim=4)
    op = DownsampleFactorMax((1, ds[0]), ignore_border)
    outtime = op(input_4D_time)
    outshape = T.join(0, input_time.shape[:-2], outtime.shape[-2:])
    shufl = (
    list(range(vid_dim - 3)) + [vid_dim - 1] + [vid_dim - 3] + [vid_dim - 2])
    return T.reshape(outtime, outshape, ndim=input.ndim).dimshuffle(shufl)
