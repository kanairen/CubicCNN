# coding:utf-8

import PIL.Image
import cPickle
import itertools
import os

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
from theano import config

from src.helper.config import path_res_2d, path_res_2d_pattern, \
    path_res_3d_shrec_target, path_res_3d_shrec_query, path_res_3d_psb, \
    path_res_3d_psb_classifier, path_res_numpy_psb_test, \
    path_res_numpy_psb_train, path_res_numpy_boxel_test, \
    path_res_numpy_boxel_train
from src.util.image import translate, distort
from src.util.parse import parse_off
from src.util.sequence import joint_dict

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


def shrec():
    target_list = os.listdir(path_res_3d_shrec_target)
    query_list = os.listdir(path_res_3d_shrec_query)
    targets = []
    queries = []
    for f_target, f_query in zip(target_list, query_list):
        target_vertices = parse_off(path_res_3d_shrec_target + f_target)[0]
        query_vertices = parse_off(path_res_3d_shrec_query + f_query)[1]
        targets.append(target_vertices)
        queries.append(query_vertices)
    return targets, queries


class PSB(object):
    @classmethod
    def load_class_info_all(cls):
        test_classes = cls.__load_class_info(is_test=True)
        train_classes = cls.__load_class_info(is_test=False)
        return joint_dict(test_classes, train_classes)

    @staticmethod
    def __load_class_info(is_test=False, train_name="train.cla",
                          test_name="test.cla"):

        classifier = {}

        f_name = test_name if is_test else train_name
        path = path_res_3d_psb_classifier + "/" + f_name

        with file(path) as f:

            line = f.readline()

            if "PSB" not in line:
                raise IOError("PSB class file must be \"cls\" format file.")
            else:
                n_class, n_model = map(int, f.readline().split(" "))

            while line:

                line = f.readline()

                if line == "\n":
                    continue

                s_line = line.split(" ")

                if len(s_line) == 3:
                    main_name, sub_name, n_id = s_line
                    n_id = int(n_id)

                    if n_id > 0:
                        ids = [int(f.readline()) for i in range(n_id)]
                        prefix = "" if sub_name == "0" else sub_name + "/"
                        classifier.setdefault(prefix + main_name, ids)

        return classifier

    @classmethod
    def load_vertices_all(cls, dir=path_res_3d_psb):
        train_vertices = []
        test_vertices = []
        train_answers = []
        test_answers = []

        train_classes = cls.__load_class_info(is_test=False)
        test_classes = cls.__load_class_info(is_test=True)
        classes = cls.load_class_info_all()

        train_members = list(itertools.chain(*train_classes.values()))
        test_members = list(itertools.chain(*test_classes.values()))

        for f in os.listdir(dir):

            if not os.path.isfile(dir + "/" + f):
                continue

            id = int(f.split(".")[0])

            if id in test_members:
                v_list = test_vertices
                a_list = test_answers
                is_test = True
            elif id in train_members:
                v_list = train_vertices
                a_list = train_answers
                is_test = False

            # データの格納
            v_list.append(cls.__load_vertices(f, dir, is_test=is_test))

            # 正解ラベルの探索と格納
            keys = classes.keys()
            for key in keys:
                if id in classes.get(key):
                    a_list.append(keys.index(key))
                    break

        return train_vertices, test_vertices, train_answers, test_answers, train_members, test_members

    @staticmethod
    def __load_vertices(f_name, dir=path_res_3d_psb, is_test=False):

        if is_test:
            np_dir = path_res_numpy_psb_test
        else:
            np_dir = path_res_numpy_psb_train

        np_path = np_dir + "/" + f_name.split(".")[0] + ".npy"

        # キャッシュがあれば、それを返す
        if os.path.exists(np_path):
            return np.load(np_path)

        vertices, faces = parse_off(dir + "/" + f_name)

        return vertices

    @staticmethod
    def __standard(points):
        mean = np.mean(points, axis=0)
        norm = np.max(points) - np.min(points)
        return np.array([(p - mean) / norm for p in points],
                        dtype=config.floatX)

    @classmethod
    def __boxel(cls, points, n_div=100):
        # -0.5~0.5
        points = cls.__standard(points)
        boxel = np.zeros(shape=(n_div, n_div, n_div), dtype=config.floatX)
        for p in points:
            x, y, z = p
            bz = int(z * n_div + n_div) / 2
            by = int(y * n_div + n_div) / 2
            bx = int(x * n_div + n_div) / 2
            boxel[bz][by][bx] = 1

        return boxel

    @classmethod
    def boxel_all(cls, points_list, n_div=100):
        return np.array([cls.__boxel(points, n_div) for points in points_list],
                        dtype=config.floatX)

    @staticmethod
    def save_vertices(vertices, id, is_test=False):
        assert vertices.shape[-1] == 3
        if is_test:
            np.save(path_res_numpy_psb_test + "/" + str(id), vertices)
        else:
            np.save(path_res_numpy_psb_train + "/" + str(id), vertices)

    @staticmethod
    def save_boxel(boxel, id, is_test):
        assert boxel.ndim == 1
        if is_test:
            np.save(path_res_numpy_boxel_test + "/" + str(id), boxel)
        else:
            np.save(path_res_numpy_boxel_train + "/" + str(id), boxel)

    @staticmethod
    def __load_boxel(id, is_test=False):
        if is_test:
            path = path_res_numpy_boxel_test + "/" + str(id) + ".npy"
        else:
            path = path_res_numpy_boxel_train + "/" + str(id) + ".npy"
        return np.load(path)

    @classmethod
    def load_boxels(cls, is_test=False):

        boxels = []
        answers = []

        classes = cls.__load_class_info(is_test=is_test)
        all_classes = cls.load_class_info_all()

        for id_list in classes.values():
            for id in id_list:
                boxels.append(cls.__load_boxel(id, is_test=is_test))
                for class_label, class_ids in enumerate(all_classes.values()):
                    if id in class_ids:
                        answers.append(class_label)
                        break

        return boxels, answers
