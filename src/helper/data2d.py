# coding:utf-8

import PIL.Image
import cPickle
import itertools
import os
import numpy as np
from collections import OrderedDict
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
from src.helper.config import path_res_2d, path_res_2d_pattern, \
    path_res_2d_psb_depth, path_res_3d_psb_classifier
from src.helper.data3d import parse_cla
from src.util.image import translate, distort

__author__ = 'ren'


def mnist(data_home=path_res_2d, is_normalized=True, is_formal=True,
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

    return x_train, x_test, y_train, y_test


def cifar10(data_home=path_res_2d, is_normalized=False, is_grayscale=False):
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


def psb_depths(ids, path=path_res_2d_psb_depth):
    ids = sorted(ids)
    # クラス情報
    cls_path = path_res_3d_psb_classifier
    train_cls = parse_cla(os.path.join(cls_path, "train.cla"))[0]
    test_cls = parse_cla(os.path.join(cls_path, "test.cla"))[0]
    all_cls = reduce(lambda x, y: OrderedDict(x, **y), (train_cls, test_cls))

    # 各データセット別IDリスト
    train_ids = sorted(list(itertools.chain(*train_cls.values())))
    test_ids = sorted(list(itertools.chain(*test_cls.values())))

    # IDから、データのクラスラベルを取得
    # クラスラベルは全クラスリスト中でのIndex
    def class_label(cls, id):
        for cls_name, cls_ids in cls.items():
            if id in cls_ids:
                return all_cls.keys().index(cls_name)
        raise IndexError("psb id:{} is not found!".format(id))

    dir_path = os.path.join(path, "{}")

    import time
    start = time.clock()

    x_train = np.r_[
        [__psb_depths_by_id(id, data_home=dir_path.format(id)) for id in ids if
         id in train_ids]]
    x_test = np.r_[
        [__psb_depths_by_id(id, data_home=dir_path.format(id)) for id in ids if
         id in test_ids]]

    y_train = np.asarray(list(itertools.chain(
        *[[class_label(train_cls, id)] * len(os.listdir(dir_path.format(id)))
          for id in ids if id in train_ids])))
    y_test = np.asarray(list(itertools.chain(
        *[[class_label(test_cls, id)] * len(os.listdir(dir_path.format(id)))
          for id in ids if id in test_ids])))

    print "process time : {}s".format(time.clock() - start)

    n_psb_train, n_img_train, w, h = x_train.shape
    n_psb_test, n_img_test = x_test.shape[:2]
    x_train = x_train.reshape((n_psb_train * n_img_train, 1, w, h))
    x_test = x_test.reshape((n_psb_test * n_img_test, 1, w, h))

    return x_train, x_test, y_train, y_test


def __psb_depths_by_id(id, data_home=path_res_2d_psb_depth):
    print id
    return np.r_[[np.load(os.path.join(data_home, file_name)) for file_name in
                  os.listdir(data_home)]]
