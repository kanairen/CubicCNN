# coding:utf-8

import six
import numpy as np
from theano import config, tensor as T, scan, printing

from src.model.layer.conv import ConvLayer2d
from src.util.activation import relu
from src.util.sequence import pair

__author__ = 'ren'


class PoolLayer(ConvLayer2d):
    POOL_MAX = 0
    POOL_AVERAGE = 1

    def __init__(self, img_size, in_channel, k_size, pad=0, dtype=config.floatX,
                 activation=relu, pool_type=POOL_MAX):

        # フィルタサイズ
        kw, kh = pair(k_size)

        # フィルタベクトル
        filter = np.ones((in_channel * in_channel * kh * kw), dtype=dtype)

        super(PoolLayer, self).__init__(img_size, in_channel, in_channel,
                                        k_size, k_size, pad, None, None, True,
                                        filter, dtype, activation)

        # 実行するプーリングの種類
        self.pool_type = pool_type

    def update(self, cost, learning_rate=0.01):
        return None

    def output(self, inputs_symbol):
        if self.pool_type == PoolLayer.POOL_MAX:
            return self.max_pooling(inputs_symbol)
        elif self.pool_type == PoolLayer.POOL_AVERAGE:
            return self.averate_pooling(inputs_symbol)
        else:
            raise RuntimeError("pool_type is invalid.")


    def max_pooling(self, inputs_symbol):
        # Wを初期化
        self.init_weight()
        # 各データに関して、プーリングの最大値とIndexを返す
        result, update = scan(self.max_args, sequences=[inputs_symbol],
                              non_sequences=self.W)
        # 最大値リストと、Indexリスト
        max_prod, args_prod = result

        # TODO args_prodがおそらくリストのリストなので、一つのリストにまとめないと怒られるかもしれない
        w_zeros = T.zeros_like(self.W)
        new_W = T.set_subtensor(w_zeros[range(self.n_out)][args_prod], 1)
        self.W.set_value(new_W)

        # TODO max_prodを返すと、逆伝播の時次元が違うと起こるかもしれない
        return max_prod


    @staticmethod
    def max_args(input, weight):
        prods = input * weight
        max_prod = T.max(prods, axis=1)
        args_prod = T.argmax(prods, axis=1)
        return max_prod, args_prod


    @staticmethod
    def max_pool_weighting(v, w_row, weight):
        max_arg = T.argmax(v)
        zeros = T.zeros_like(v)
        weight[w_row].fill(0)

        return T.set_subtensor(zeros[max_arg], v[max_arg])
