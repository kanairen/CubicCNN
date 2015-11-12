# coding:utf-8

import os
import itertools
import cPickle
import numpy as np
import PIL.Image
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from src.util.config import path_res_2d, path_res_2d_pattern

__author__ = 'ren'


def mnist(data_home=path_res_2d, test_size=0.2, is_normalized=False,
          x_dtype=np.float32, y_dtype=np.int32):
    # MNIST手書き文字データ data_homeのデータを読み込む
    # データがない場合はWEB上からダウンロード
    mnist = fetch_mldata('MNIST original', data_home=data_home)

    x = mnist.data.astype(x_dtype)
    y = mnist.target.astype(y_dtype)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=test_size)

    x_test = x_test.reshape(len(x_test), 1, 24, 24)
    x_train = x_train.reshape(len(x_train, 1, 24, 24))

    return x_train, x_test, y_train, y_test


def cifar10(data_home=path_res_2d):
    def unpickle(file):
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict

    path = data_home + "/cifar-10-batches-py"

    x_train = None
    y_train = []

    for i in xrange(1, 6):
        data_dictionary = unpickle(path + "/data_batch_%d" % i)
        if x_train == None:
            x_train = data_dictionary['data']
        else:
            x_train = np.vstack((x_train, data_dictionary['data']))
        y_train = y_train + data_dictionary['labels']

    test_data_dictionary = unpickle(path + "/test_batch")
    x_test = test_data_dictionary['data']
    x_test = x_test.reshape(len(x_test), 3, 32, 32)
    y_train = np.array(y_train)
    x_train = x_train.reshape((len(x_train), 3, 32, 32))
    y_test = np.array(test_data_dictionary['labels'])

    return x_train, x_test, y_train, y_test


