#!/usr/bin/env python
# coding: utf-8

import re
import shutil
import numpy as np
import tqdm
import warnings
from CubicCNN import *
from CubicCNN.src.util import parseutil
from __cls import DataLoader, Data3d


class PSBLoader(DataLoader):
    def __init__(self, archive_home=PATH_RES_TMP):
        data_home, root_name = os.path.split(PATH_RES_SHAPE_PSB)
        super(PSBLoader, self).__init__(data_home, archive_home, root_name)

    def load(self, url=PSB_DATASET_URL):

        super(PSBLoader, self)._load(url, self.root_name)

        prev_path = os.path.join(self.archive_home, self.root_name)
        prev_cla_path = os.path.join(prev_path, PSB_RELATIVE_CLA_PATH)
        prev_off_path = os.path.join(prev_path, PSB_RELATIVE_OFF_PATH)

        # .cla
        if not os.path.exists(PATH_RES_SHAPE_PSB_CLASS):
            os.makedirs(PATH_RES_SHAPE_PSB_CLASS)
        if not os.path.exists(PATH_RES_SHAPE_PSB_OFF):
            os.makedirs(PATH_RES_SHAPE_PSB_OFF)

        shutil.copy(os.path.join(prev_cla_path, PSB_CLS_TEST),
                    os.path.join(PATH_RES_SHAPE_PSB_CLASS, PSB_CLS_TEST))
        shutil.copy(os.path.join(prev_cla_path, PSB_CLS_TRAIN),
                    os.path.join(PATH_RES_SHAPE_PSB_CLASS, PSB_CLS_TRAIN))

        # .off
        for root, dirnames, filenames in os.walk(prev_off_path):
            for filename in filenames:
                if ".off" in filename:
                    shutil.copy(os.path.join(root, filename),
                                os.path.join(PATH_RES_SHAPE_PSB_OFF, filename))


def psb_voxel(is_co_class=False, is_cached=False, from_cached=False):
    if from_cached:
        print 'load voxels from .npy ...'
        path = PATH_RES_SHAPE_PSB_BINVOX_CACHE_CO_CLASS \
            if is_co_class else PATH_RES_SHAPE_PSB_BINVOX_CACHE_DEFAULT
        try:
            return Data3d.load(path)
        except IOError:
            warnings.warn('psb_voxel data was not loaded.')

    print 'load voxels from .off ...'

    cla_train = parseutil.parse_cla(
        os.path.join(PATH_RES_SHAPE_PSB_CLASS, PSB_CLS_TRAIN))
    cla_test = parseutil.parse_cla(
        os.path.join(PATH_RES_SHAPE_PSB_CLASS, PSB_CLS_TEST))
    if is_co_class:
        cla_list = list(
            set(cla_train.keys()).intersection(set(cla_test.keys())))
    else:
        cla_list = list(set(cla_train.keys() + cla_test.keys()))

    re_compile = re.compile("\d+")

    def check_binvox_y(binvox_id, is_train):
        cla = cla_train if is_train else cla_test
        for c_label, c_ids in cla.items():
            if binvox_id in c_ids:
                try:
                    return cla_list.index(c_label)
                except ValueError:
                    break
        return None

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for f in tqdm.tqdm(os.listdir(PATH_RES_SHAPE_PSB_BINVOX)):
        binvox_path = os.path.join(PATH_RES_SHAPE_PSB_BINVOX, f)
        if os.path.isdir(binvox_path):
            continue
        binvox = parseutil.parse_binvox(binvox_path)
        binvox_id = int(re_compile.findall(f)[0])
        train_binvox_y = check_binvox_y(binvox_id, True)
        if train_binvox_y:
            x_train.append(binvox)
            y_train.append(train_binvox_y)
        else:
            test_binvox_y = check_binvox_y(binvox_id, False)
            if test_binvox_y:
                x_test.append(binvox)
                y_test.append(test_binvox_y)
            else:
                continue

    x_train = np.asarray(x_train, dtype=np.uint8)
    x_test = np.asarray(x_test, dtype=np.uint8)
    y_train = np.asarray(y_train, dtype=np.uint8)
    y_test = np.asarray(y_test, dtype=np.uint8)

    x_train = x_train.reshape([x_train.shape[0], 1] + list(x_train.shape[1:]))
    x_test = x_test.reshape([x_test.shape[0], 1] + list(x_test.shape[1:]))

    data = Data3d(x_train, x_test, y_train, y_test)

    if is_cached:
        save_path = PATH_RES_SHAPE_PSB_BINVOX_CACHE_CO_CLASS \
            if is_co_class else PATH_RES_SHAPE_PSB_BINVOX_CACHE_DEFAULT
        data.save(save_path)

    return data
