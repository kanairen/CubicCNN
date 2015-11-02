# coding:utf-8

import six
import numpy as np

__author__ = 'ren'


class FilterLayer(object):
    def __init__(self):
        pass

    @staticmethod
    def init_T(img_w, img_h, kw, kh, sw, sh, n_in, n_out, in_channel,
               out_channel, dtype):

        filter_length = out_channel * in_channel * kh * kw

        # 重みnumpy行列
        T = np.zeros(shape=(n_out, n_in, filter_length), dtype=dtype)

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
