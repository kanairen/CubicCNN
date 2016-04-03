# coding: utf-8

import numpy as np
from theano import config, shared
from theano.tensor.nnet.conv import conv2d
from theano.sandbox.cuda.fftconv import conv2d_fft
from __grid import GridLayer2d


class ConvLayer2d(GridLayer2d):
    def __init__(self, layer_id, image_size, activation, c_in, c_out, k,
                 s=(1, 1), p=(0, 0), filters=None, b=None, border_mode='valid',
                 dtype=config.floatX, is_dropout=False, dropout_rate=0.5):

        super(ConvLayer2d, self).__init__(layer_id, image_size, c_in, c_out, k,
                                          s, p, activation, is_dropout,
                                          dropout_rate)

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

        self.params = self.filters, self.params

        self.border_mode = border_mode

    def output(self, input, is_train):

        if input.ndim != 4:
            input = input.reshape(
                (input.shape[0], self.c_in, self.image_size[0],
                 self.image_size[1]))

        u = conv2d(input, self.filters,
                   filter_shape=(self.c_out, self.c_in, self.k[1], self.k[0]),
                   border_mode=self.border_mode,
                   subsample=self.s)
        # TODO バイアスの挿入位置がここでほんとうに正しいのかテスト
        return self._activate(u + self.b, is_train)

    def __str__(self):
        return super(ConvLayer2d, self).__str__()
