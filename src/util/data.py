# coding:utf-8

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from src.util.config import path_res_2d

__author__ = 'ren'


def mnist(data_home=path_res_2d, test_size=0.2, is_normalized=False,
          x_dtype=np.float32, y_dtype=np.int32):
    # MNIST手書き文字データ data_homeのデータを読み込む
    # データがない場合はWEB上からダウンロード
    mnist = fetch_mldata('MNIST original', data_home=data_home)

    x = mnist.data.astype(x_dtype)
    y = mnist.target.astype(y_dtype)

    return train_test_split(x, y, test_size=test_size)

