# coding:utf-8

import os
import numpy as np
import itertools
import recognizer
import enum
from src.client.structure.shape_structure import cnn
from src.helper.data3d import psb_binvoxs, psb_ids
from src.helper.config import path_res_numpy_cache_psb
from src.util.shape import trio, rotate_voxels, trans_voxels, \
    centerize_voxel

__author__ = 'ren'

AUG_TYPE = enum.Enum("AUG_TYPE", "AUG_ROTATE AUG_TRANSLATE")

FILE_NP_CACHE_X_TRAIN = "x_train_{}_{}_f{}_t{}.npy"
FILE_NP_CACHE_X_TEST = "x_test_{}_{}_f{}_t{}.npy"
FILE_NP_CACHE_Y_TRAIN = "y_train_{}_{}_f{}_t{}.npy"
FILE_NP_CACHE_Y_TEST = "y_test_{}_{}_f{}_t{}.npy"


def shape_recognition(data_type, n_iter, n_batch, aug_type,
                      box=(100, 100, 100), from_r=(0, 0, -10), to_r=(1, 1, 10),
                      step=2, show_batch_accuracies=False,
                      save_batch_accuracies=False, load_voxels=False,
                      save_voxels=False):
    print "box : ", box
    print "from_r : ", from_r
    print "to_r : ", to_r
    print "step : ", step

    if data_type == "psb_binvox":
        ids = psb_ids([], is_all=True, is_both=True)
        psb_binvox_recognition(ids, n_iter, n_batch, aug_type, box, from_r,
                               to_r, step, show_batch_accuracies,
                               save_batch_accuracies, load_voxels, save_voxels)


def psb_binvox_recognition(ids, n_iter, n_batch, aug_type, box, from_r, to_r,
                           step, show_batch_accuracies=False,
                           save_batch_accuracies=False,
                           load_voxels=False, save_voxels=False):
    f_x_train = FILE_NP_CACHE_X_TRAIN.format(box, aug_type, from_r, to_r)
    f_x_test = FILE_NP_CACHE_X_TEST.format(box, aug_type, from_r, to_r)
    f_y_train = FILE_NP_CACHE_Y_TRAIN.format(box, aug_type, from_r, to_r)
    f_y_test = FILE_NP_CACHE_Y_TEST.format(box, aug_type, from_r, to_r)

    if load_voxels:
        x_train = np.load(os.path.join(path_res_numpy_cache_psb, f_x_train))
        x_test = np.load(os.path.join(path_res_numpy_cache_psb, f_x_test))
        y_train = np.load(os.path.join(path_res_numpy_cache_psb, f_y_train))
        y_test = np.load(os.path.join(path_res_numpy_cache_psb, f_y_test))

    else:

        print "the number of 3D models : ", len(ids)

        x_train, x_test, y_train, y_test = psb_binvoxs(ids)

        n_in = reduce(lambda x, y: x * y, box)
        n_r = reduce(lambda x, y: x * y,
                     ((t - f) / s if (t - f) / s >= 1 else 1 for f, t, s in
                      zip(from_r, to_r, trio(step))))

        r_x_train = []
        r_x_test = []

        # ボクセルの中心
        center = (box[0] / 2, box[1] / 2, box[2] / 2)

        for i, data in enumerate(zip(x_train, x_test)):
            print "{}th data being created..".format(i + 1)
            train, test = data
            c_train = centerize_voxel(train, center)
            c_test = centerize_voxel(test, center)
            if aug_type == AUG_TYPE.AUG_ROTATE.name:
                train_voxels = rotate_voxels(c_train, from_r, to_r, step,
                                             center)
                test_voxels = rotate_voxels(c_test, from_r, to_r, step, center)
            elif aug_type == AUG_TYPE.AUG_TRANSLATE.name:
                train_voxels = trans_voxels(c_train, from_r, to_r, step)
                test_voxels = trans_voxels(c_train, from_r, to_r, step)
            else:
                train_voxels = [c_train, ]
                test_voxels = [c_test, ]
            r_x_train.extend(train_voxels)
            r_x_test.extend(test_voxels)

        print "reshape..."
        x_train = np.asarray(r_x_train).reshape(len(r_x_train), n_in)
        x_test = np.asarray(r_x_test).reshape(len(r_x_test), n_in)
        y_train = np.asarray(
                list(itertools.chain(*[[y] * n_r for y in y_train])))
        y_test = np.asarray(list(itertools.chain(*[[y] * n_r for y in y_test])))

        if save_voxels:
            np.save(os.path.join(path_res_numpy_cache_psb, f_x_train), x_train)
            np.save(os.path.join(path_res_numpy_cache_psb, f_x_test), x_test)
            np.save(os.path.join(path_res_numpy_cache_psb, f_y_train), y_train)
            np.save(os.path.join(path_res_numpy_cache_psb, f_y_test), y_test)

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
