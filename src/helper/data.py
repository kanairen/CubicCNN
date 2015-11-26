# coding:utf-8

import PIL.Image
import cPickle
import itertools
import os
import collections
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
from theano import config
from src.helper.config import path_res_2d, path_res_2d_pattern, \
    path_res_3d_shrec_target, path_res_3d_shrec_query, path_res_3d_psb, \
    path_res_3d_psb_classifier, path_res_numpy_psb_test, \
    path_res_numpy_psb_train, path_res_numpy_boxel_psb_test, \
    path_res_numpy_boxel_psb_train
from src.util.image import translate, distort
from src.util.parse import parse_off, parse_cla

__author__ = 'ren'


def mnist(data_home=path_res_2d, test_size=0.2, is_normalized=True,
          x_dtype=np.float32, y_dtype=np.int32):
    # MNIST手書き文字データ data_homeのデータを読み込む
    # データがない場合はWEB上からダウンロード
    mnist = fetch_mldata('MNIST original', data_home=data_home)

    x = mnist.data.astype(x_dtype)
    y = mnist.target.astype(y_dtype)

    if is_normalized:
        x /= x.max()

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=test_size)
    x_test = x_test.reshape((len(x_test), 1, 28, 28))
    x_train = x_train.reshape((len(x_train), 1, 28, 28))

    return x_train, x_test, y_train, y_test


def cifar10(data_home=path_res_2d, is_normalized=True, is_grayscale=False,
            x_dtype=np.float32, y_dtype=np.int32):
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
        if x_train is None:
            x_train = data_dictionary['data']
        else:
            x_train = np.vstack((x_train, data_dictionary['data']))
        y_train = y_train + data_dictionary['labels']

    test_data_dictionary = unpickle(path + "/test_batch")

    x_test = test_data_dictionary['data']
    y_train = np.array(y_train)
    y_test = np.array(test_data_dictionary['labels'])

    x_test = x_test.reshape(len(x_test), 3, 32, 32)
    x_train = x_train.reshape((len(x_train), 3, 32, 32))

    if is_grayscale:
        c, h, w = x_test.shape[1:]
        x_test_mean = np.zeros((len(x_test), h, w))
        x_train_mean = np.zeros((len(x_train), h, w))
        for i in xrange(c):
            x_test_mean += x_test[:, i, :, :]
            x_train_mean += x_train[:, i, :, :]
        x_test_mean /= c
        x_train_mean /= c
        x_test = x_test_mean.reshape((len(x_test), 1, 32, 32))
        x_train = x_train_mean.reshape((len(x_train), 1, 32, 32))

    if is_normalized:
        x_test /= x_test.max()
        x_train /= x_train.max()

    x_test = x_test.astype(x_dtype)
    x_train = x_train.astype(x_dtype)
    y_test = y_test.astype(y_dtype)
    y_train = y_train.astype(y_dtype)

    return x_train, x_test, y_train, y_test


def pattern50_rotate(img_size=None, is_binary=True, is_flatten=False,
                     data_home=path_res_2d_pattern,
                     test_size=0.2, rotate_angle=20, step=1, dtype=np.int8):
    return __pattern50(img_size, 'rotate', is_binary, is_flatten, data_home,
                       test_size, rotate_angle=rotate_angle, step=step,
                       dtype=dtype)


def pattern50_trans(img_size=None, is_binary=True, is_flatten=False,
                    data_home=path_res_2d_pattern,
                    test_size=0.2, trans_x=(-4, 4), trans_y=(-5, 5),
                    dtype=np.int8):
    return __pattern50(img_size, 'trans', is_binary, is_flatten, data_home,
                       test_size, trans_x=trans_x, trans_y=trans_y, dtype=dtype)


def pattern50_distort(img_size=None, is_binary=True, is_flatten=False,
                      data_home=path_res_2d_pattern, test_size=0.2,
                      distorted_size=(64, 64), n_images=20, fix_distort=False,
                      dtype=np.int8):
    return __pattern50(img_size, 'distort', is_binary, is_flatten, data_home,
                       test_size, distorted_size=distorted_size,
                       n_images=n_images, fix_distort=fix_distort, dtype=dtype)


def __pattern50(img_size, process, is_binary, is_flatten,
                data_home=path_res_2d_pattern, test_size=0.2, step=1,
                rotate_angle=20, trans_x=(-4, 4), trans_y=(-5, 5),
                distorted_size=(64, 64), n_images=4, fix_distort=False,
                dtype=np.int8):
    # 入力データリスト
    x = []
    # 正解データリスト
    y = []

    # 指定された変換を画像に適用し、変換結果をリストで受け取る

    for i, f_name in enumerate(os.listdir(data_home)):
        # 元画像
        image = PIL.Image.open(data_home + "/" + f_name)

        # 二値画像に変換
        if is_binary:
            image = image.convert('1')

        # 画像サイズ変更
        if img_size is not None:
            image = image.resize(img_size)

        # 処理後の
        if process == 'rotate':
            images = __rotate_images(image, rotate_angle, step)
        elif process == 'trans':
            images = __translate_images(image, trans_x, trans_y, step)
        elif process == 'distort':
            images = __distort_images(image, distorted_size, n_images,
                                      fix_distort)
        else:
            images = [image]

        if is_flatten:
            images = [np.asarray(img).astype(dtype).flatten() for img in images]
        else:
            images = [np.asarray(img).astype(dtype) for img in images]

        # 入力リストに画像データを追加
        x.extend(images)
        # 正解リストに正解ラベルを追加
        y.extend([i] * len(images))

    x = np.array(x, dtype=dtype)
    y = np.array(y, dtype=np.int32)

    if is_binary and not is_flatten:
        x = x.reshape((len(x), 1, x.shape[1], x.shape[2]))

    return train_test_split(x, y, test_size=test_size)


def __rotate_images(image, angle, step):
    return [image.rotate(r) for r in range(angle)]


def __translate_images(image, trans_x, trans_y, step):
    assert len(trans_x) == 2
    assert len(trans_y) == 2

    # 画像の平行移動
    t_range_x = range(trans_x[0], trans_x[1], step)
    t_range_y = range(trans_y[0], trans_y[1], step)
    t_imgs_2d = [[translate(image, w, h) for w in t_range_x] for h in
                 t_range_y]
    return list(itertools.chain(*t_imgs_2d))


def __distort_images(image, newsize, n_images=4, fix_distort=False):
    # 圧縮前の画像サイズ
    w, h = image.size
    # 圧縮後の画像サイズ
    new_w, new_h = newsize

    assert w > new_w and h > new_h

    rnd = np.random.RandomState()

    if fix_distort:
        seeds = np.random.randint(low=0, high=9999, size=(n_images,))

    images = []

    for k in xrange(n_images):
        if fix_distort:
            rnd.seed(seeds[k])
        distorted_image = distort(image, newsize, rnd=rnd)
        images.append(distorted_image)

    return images


"""
3D
"""



