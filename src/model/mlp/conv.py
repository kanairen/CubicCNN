# coding:utf-8

import numpy as np
from theano import config, shared
from layer import Layer, rnd

__author__ = 'ren'


# TODO 順伝播 30min
# TODO　逆電波 30min

# TODO 畳み込み 1h
# TODO プーリング 1h

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return (x, x)


class ConvLayer2d(Layer):
    def __init__(self, img_size, k_size, in_channel, out_channel,
                 W=None, b=None, dtype=config.floatX, activation=None):
        super(ConvLayer2d, self).__init__(n_in, n_out, W=W, b=b, dtype=dtype,
                                          activation=activation)
        # 画像サイズ
        w, h = _pair(img_size)

        # フィルタサイズ
        kw, kh = _pair(k_size)

        # 入力・出力ユニット数
        n_in = w * h * in_channel
        n_out = w * h * out_channel

        # 共有重み抽出ベクトル
        T = shared(np.zeros(
            shape=(n_in * n_out, kw * kh * in_channel * out_channel),
            dtype=dtype
        ), name='T', borrow=True)

        # フィルタベクトル
        h = shared(np.asarray(
            rnd.uniform(low=-np.sqrt(1. / in_channel * kw * kh),
                        high=np.sqrt(1. / in_channel * kw * kh),
                        size=(kw * kh * in_channel * out_channel)),
            dtype=dtype
        ), name='W', borrow=True)

        if W is None:
            W = shared(np.zeros(shape=(n_in, n_out)), name='W', borrow=True)

        def output(self):
            pass
