# coding:utf-8

import os
import numpy as np
import itertools
import recognizer
from src.client.structure.shape_structure import cnn
from src.helper.data3d import psb_binvoxs, psbs, voxelize_all
from src.helper.config import path_res_numpy_cache_psb
from src.util.shape import rotate_shapes, rotate_voxels, trans_voxels, \
    centerize_voxel

__author__ = 'ren'


def shape_recognition(data_type, n_iter, n_batch, show_batch_accuracies=False,
                      save_batch_accuracies=False,
                      box=(100, 100, 100), r=(1, 1, 45), step=5):
    if data_type == "psb":
        psb_recognition(n_iter, n_batch, box, r, step, show_batch_accuracies,
                        save_batch_accuracies)
    elif data_type == "psb_binvox":
        psb_binvox_recognition(n_iter, n_batch, box, r, step,
                               show_batch_accuracies, save_batch_accuracies)


def psb_binvox_recognition(n_iter, n_batch, box, r, step,
                           show_batch_accuracies=False,
                           save_batch_accuracies=False):
    # PSB 3DモデルのID
    ids = []
    # airplane biplane
    ids.extend(xrange(1118, 1145 + 1, 1))
    # airplane commercial(test -1)
    ids.extend(xrange(1146, 1166, 1))
    # fright jet
    ids.extend(xrange(1167, 1266 + 1))
    # helicopter(test -1)
    ids.extend(xrange(1302, 1336))
    # enterprise_like
    ids.extend(xrange(1353, 1374 + 1))

    x_train, x_test, y_train, y_test = psb_binvoxs(ids)

    n_in = reduce(lambda x, y: x * y, box)
    n_r = reduce(lambda x, y: x * y, r)

    r_x_train = []
    r_x_test = []

    for i, data in enumerate(zip(x_train, x_test)):
        train, test = data
        c_train = centerize_voxel(train, (50, 50, 50))
        c_test = centerize_voxel(test, (50, 50, 50))
        r_x_train.extend(rotate_voxels(c_train, r, step, (50, 50, 50)))
        r_x_test.extend(rotate_voxels(c_test, r, step, (50, 50, 50)))

    print "reshape..."
    x_train = np.asarray(r_x_train).reshape(len(r_x_train), n_in)
    x_test = np.asarray(r_x_test).reshape(len(r_x_test), n_in)
    y_train = np.asarray(list(itertools.chain(*[[y] * n_r for y in y_train])))
    y_test = np.asarray(list(itertools.chain(*[[y] * n_r for y in y_test])))

    print "train data : ", len(x_train)
    print "test data : ", len(x_test)
    print "classes : ", len(set(y_train.tolist() + y_test.tolist()))

    print "preparing models..."

    model = cnn(box)

    """
    # TRAIN
    # """

    recognizer.learning(model, x_train, x_test, y_train, y_test, n_iter,
                        n_batch, show_batch_accuracies, save_batch_accuracies,
                        is_batch_test=True)


def psb_recognition(n_iter, n_batch, box, r, step, show_batch_accuracies=False,
                    save_batch_accuracies=False):
    ids = []
    # airplane biplane
    ids.extend(xrange(1118, 1145, 1))
    # airplane commercial
    ids.extend(xrange(1146, 1166, 1))

    x_train, x_test, y_train, y_test = psbs(ids)

    # x_train = np.load(os.path.join(path_res_numpy_cache_psb,"x_train_{}.npy".format(box_size)))
    # x_test = np.load(os.path.join(path_res_numpy_cache_psb,"x_test_{}.npy".format(box_size)))
    # y_train = np.load(os.path.join(path_res_numpy_cache_psb,"y_train_{}.npy".format(box_size))).astype(np.int8)
    # y_test = np.load(os.path.join(path_res_numpy_cache_psb,"y_test_{}.npy".format(box_size))).astype(np.int8)

    r_x_train = []
    r_x_test = []

    for train, test in zip(x_train, x_test):
        r_x_train.extend(rotate_shapes(train, r, step))
        r_x_test.extend(rotate_shapes(test, r, step))

    print "boxel..."
    # ボクセル化
    x_train = voxelize_all(r_x_train)
    x_test = voxelize_all(r_x_test)

    n_in = reduce(lambda x, y: x * y, box)
    n_r = reduce(lambda x, y: x * y, r)

    print "reshape..."
    x_train = np.asarray(x_train).reshape(len(x_train), n_in)
    x_test = np.asarray(x_test).reshape(len(x_test), n_in)
    y_train = np.asarray(list(itertools.chain(*[[y] * n_r for y in y_train])))
    y_test = np.asarray(list(itertools.chain(*[[y] * n_r for y in y_test])))

    np.save(os.path.join(path_res_numpy_cache_psb, "x_train_{}".format(box)),
            x_train)
    np.save(os.path.join(path_res_numpy_cache_psb, "x_test_{}".format(box)),
            x_test)
    np.save(os.path.join(path_res_numpy_cache_psb, "y_train_{}".format(box)),
            y_train)
    np.save(os.path.join(path_res_numpy_cache_psb, "y_test_{}".format(box)),
            y_test)

    print "train data : ", len(x_train)
    print "test data : ", len(x_test)
    print "classes : ", len(set(y_train.tolist() + y_test.tolist()))

    print "preparing models..."

    model = cnn(box)

    """
    # TRAIN
    # """

    recognizer.learning(model, x_train, x_test, y_train, y_test, n_iter,
                        n_batch, show_batch_accuracies, save_batch_accuracies,
                        is_batch_test=True)
