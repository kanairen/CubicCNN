#!/usr/bin/env python
# coding: utf-8

import re
import shutil
import numpy as np
import tqdm
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


def psb_voxel(data_path=PATH_RES_SHAPE_PSB_BINVOX, dtype=np.float32,
              is_co_class=False):
    # TODO yield方式にする

    cla_train = parseutil.parse_cla(
        os.path.join(PATH_RES_SHAPE_PSB_CLASS, PSB_CLS_TRAIN))
    cla_test = parseutil.parse_cla(
        os.path.join(PATH_RES_SHAPE_PSB_CLASS, PSB_CLS_TEST))
    if is_co_class:
        cla_list = list(
            set(cla_train.keys()).intersection(set(cla_test.keys())))
    else:
        cla_list = list(set(cla_train.keys() + cla_test.keys()))

    x_train = []
    x_test = []
    y_train = []
    y_test = []

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

    for f in tqdm.tqdm(os.listdir(data_path)):
        binvox = parseutil.parse_binvox(os.path.join(data_path, f))
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

    x_train = np.asarray(x_train, dtype=dtype)
    x_test = np.asarray(x_test, dtype=dtype)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)

    x_train = x_train.reshape([x_train.shape[0], 1] + list(x_train.shape[1:]))
    x_test = x_test.reshape([x_test.shape[0], 1] + list(x_test.shape[1:]))

    return Data3d(x_train, x_test, y_train, y_test)