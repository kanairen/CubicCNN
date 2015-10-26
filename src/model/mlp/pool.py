# coding:utf-8

import numpy as np
import six
from layer import Layer
from theano import config
from src.util.conv import pair
from src.util.activation import relu

__author__ = 'ren'


class PoolLayer(object):
    POOL_MAX = 0
    POOL_AVERAGE = 1

    def __init__(self, img_size, k_size, in_channel, stride=1, pad=0, W=None,
                 b=None, dtype=config.floatX, activation=None):

        # 入力画像のサイズ
        self.img_w, self.img_h = pair(img_size)

        # プーリング領域の大きさ
        self.kw, self.kh = pair(k_size)

        # ストライド
        self.sw, self.sh = pair(stride)

        # パディング
        self.pw, self.ph = pair(pad)

        # 入力チャンネル
        self.in_channel = in_channel

        # 活性化関数
        if activation is None:
            activation = relu
        self.activation = activation


    def pool(self, img):
        output = []
        # フィルタ移動
        for c in six.moves.range(self.in_channel):
            for h in six.moves.range(0, self.img_h, self.sh):
                for w in six.moves.range(0, self.img_w, self.sw):
                    # プーリング対象
                    pool_set = set()
                    #
                    for kh in six.moves.range(self.kh):
                        for kw in six.moves.range(self.kw):
                            pool_set.add(img[c][h + kh][w + kw])
                    output.append(max(pool_set))
