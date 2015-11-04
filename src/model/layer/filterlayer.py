# coding:utf-8

import six
import numpy as np
from theano import config, shared
from src.util.sequence import pair
from src.interface.layerinterface import LayerInterface

__author__ = 'ren'


class FilterLayer(LayerInterface):
    def __init__(self, img_size, in_channel, out_channel, k_size, stride=1,
                 T=None, b=None, no_bias=False, h=None,
                 dtype=config.floatX, activation=None):

        # 乱数生成器
        self.rnd = np.random.RandomState(1111)

        # 画像サイズ
        img_w, img_h = pair(img_size)

        # フィルタサイズ
        kw, kh = pair(k_size)

        # ストライド
        sw, sh = pair(stride)

        # 入力・出力ユニット数
        n_in = img_w * img_h * in_channel
        n_out = img_w * img_h * out_channel / (stride ** 2)

        # 重み・フィルタ変換行列
        if T is None:
            T = self.init_T(img_w, img_h, kw, kh, sw, sh, n_in, n_out,
                            in_channel, out_channel, dtype=dtype)
        self.T = shared(T, name='T', borrow=True)

        # フィルタベクトル
        if h is None:
            h = np.asarray(
                self.rnd.uniform(low=-np.sqrt(1. / in_channel * kw * kh),
                                 high=np.sqrt(1. / in_channel * kw * kh),
                                 size=(out_channel * in_channel * kh * kw)),
                dtype=dtype)
        self.h = shared(h, name='filter', borrow=True)

        # バイアスベクトル
        if not no_bias:
            if b is None:
                b = np.zeros(shape=(n_out,), dtype=dtype)
            self.b = shared(b, name='b', borrow=True)

        # 活性化関数
        if activation is None:
            activation = lambda x: x
        self.activation = activation

        # 更新対象パラメタ
        self.params = [self.h, ]
        if not no_bias:
            self.params.append(self.b)

    @staticmethod
    def init_T(img_w, img_h, kw, kh, sw, sh, n_in, n_out, in_channel,
               out_channel, dtype):

        # 重みnumpy行列
        T = np.zeros(shape=(n_out, n_in, out_channel * in_channel * kh * kw),
                     dtype=dtype)

        max_w = img_w - kw
        max_h = img_h - kh

        ksq = kw * kh
        kcsq = out_channel * ksq

        for in_c in six.moves.range(in_channel):

            for in_j in six.moves.range(max_h):

                for in_i in six.moves.range(max_w):

                    for out_c in six.moves.range(out_channel):

                        for out_j in six.moves.range(0, max_h, sh):

                            for out_i in six.moves.range(0, max_w, sw):

                                j = out_c * img_w * img_h + out_j * img_w + out_i
                                i = in_c * img_w * img_h + in_j * img_w + in_i

                                k_w = out_i - in_i
                                k_h = out_j - in_j

                                if 0 <= k_w < kw and 0 <= k_h < kh:
                                    T[j][i][
                                        in_c * kcsq + out_c * ksq + k_h * kw + k_w] = 1.

        return T