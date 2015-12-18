# coding:utf-8

import os
import numpy as np
import itertools
import recognizer
from src.client.structure.solid_structure import cnn
from src.helper.data3d import rotate_shapes, psbs, boxel_all
from src.helper.config import path_res_numpy_cache_psb

__author__ = 'ren'


def psb_recognition(n_iter, n_batch, show_batch_accuracies=False,
                    save_batch_accuracies=False):

    ids = []
    # airplane biplane
    ids.extend(xrange(1118, 1145, 1))
    # airplane commercial
    ids.extend(xrange(1146, 1166, 1))

    x_train, x_test, y_train, y_test = psbs(ids)

    box_size = (100, 100, 100)

    # x_train = np.load(os.path.join(path_res_numpy_cache_psb,"x_train_{}.npy".format(box_size)))
    # x_test = np.load(os.path.join(path_res_numpy_cache_psb,"x_test_{}.npy".format(box_size)))
    # y_train = np.load(os.path.join(path_res_numpy_cache_psb,"y_train_{}.npy".format(box_size))).astype(np.int8)
    # y_test = np.load(os.path.join(path_res_numpy_cache_psb,"y_test_{}.npy".format(box_size))).astype(np.int8)

    r = (25, 25, 5)
    step = 5
    r_x_train = []
    r_x_test = []

    for train, test in zip(x_train, x_test):
        r_x_train.extend(rotate_shapes(train, r, step))
        r_x_test.extend(rotate_shapes(test, r, step))

    print "boxel..."
    # ボクセル化
    x_train = boxel_all(r_x_train)
    x_test = boxel_all(r_x_test)

    n_in = reduce(lambda x, y: x * y, box_size)
    n_r = reduce(lambda x, y: x * y, r)

    print "reshape..."
    x_train = np.asarray(x_train).reshape(len(x_train), n_in)
    x_test = np.asarray(x_test).reshape(len(x_test), n_in)
    y_train = np.asarray(list(itertools.chain(*[[y] * n_r for y in y_train])))
    y_test = np.asarray(list(itertools.chain(*[[y] * n_r for y in y_test])))

    np.save(os.path.join(path_res_numpy_cache_psb,"x_train_{}".format(box_size)),x_train)
    np.save(os.path.join(path_res_numpy_cache_psb,"x_test_{}".format(box_size)),x_test)
    np.save(os.path.join(path_res_numpy_cache_psb,"y_train_{}".format(box_size)),y_train)
    np.save(os.path.join(path_res_numpy_cache_psb,"y_test_{}".format(box_size)),y_test)

    print "train data : ", len(x_train)
    print "test data : ", len(x_test)
    print "classes : ", len(set(y_train.tolist() + y_test.tolist()))

    print "preparing models..."

    model = cnn(box_size)

    """
    # TRAIN
    # """

    recognizer.learning(model, x_train, x_test, y_train, y_test, n_iter,
                        n_batch, show_batch_accuracies, save_batch_accuracies,is_batch_test=True)
