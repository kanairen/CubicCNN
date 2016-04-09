#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata

from CubicCNN import PATH_RES_IMAGE
from __cls import Data2d


def mnist(data_home=PATH_RES_IMAGE, is_normalized=True, is_formal=True,
          test_size=0.2, x_dtype=np.float32, y_dtype=np.int32):
    # MNIST手書き文字データ data_homeのデータを読み込む
    # データがない場合はWEB上からダウンロード
    mnist = fetch_mldata('MNIST original', data_home=data_home)

    x = mnist.data.astype(x_dtype)
    y = mnist.target.astype(y_dtype)

    if is_normalized:
        x /= x.max()

    if is_formal:
        # MNISTベンチマークと同じ形式
        n_tr = 60000
        x_train, x_test, y_train, y_test = x[:n_tr], x[n_tr:], y[:n_tr], y[
                                                                         n_tr:]
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=test_size)
    x_test = x_test.reshape((len(x_test), 1, 28, 28))
    x_train = x_train.reshape((len(x_train), 1, 28, 28))

    return Data2d(x_train, x_test, y_train, y_test)
