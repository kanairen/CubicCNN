#!/usr/bin/env python
# coding: utf-8

import re
import shutil
import numpy as np
import tqdm
import warnings
from sklearn.cross_validation import train_test_split
from CubicCNN import *
from CubicCNN.src.util.parseutil import parse_cla, parse_binvox, parse_vxl
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


class PSBVoxel(Data3d):
    def __init__(self, x_train, x_test, y_train, y_test):
        super(PSBVoxel, self).__init__(x_train, x_test, y_train, y_test)

    @classmethod
    def load(cls, path):
        d = super(PSBVoxel, cls).load(path)
        return PSBVoxel(d.x_train, d.x_test, d.y_train, d.y_test)

    @staticmethod
    def create(is_co_class=False, is_cached=False, from_cached=False,
               align_data=False):
        if from_cached:
            print 'load voxels from .npy ...'
            if is_co_class:
                load_path = PATH_RES_SHAPE_PSB_BINVOX_CACHE_CO_CLASS_ORIGIN
            else:
                load_path = PATH_RES_SHAPE_PSB_BINVOX_CACHE_DEFAULT_ORIGIN
            try:
                return PSBVoxel.load(load_path)
            except IOError:
                warnings.warn('psb_voxel data was not loaded.')

        print 'load voxels from .off ...'

        cla_train = parse_cla(
            os.path.join(PATH_RES_SHAPE_PSB_CLASS, PSB_CLS_TRAIN))
        cla_test = parse_cla(
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
            binvox = parse_binvox(binvox_path)
            binvox_id = int(re_compile.findall(f)[0])
            train_binvox_y = check_binvox_y(binvox_id, True)
            if train_binvox_y is not None:
                x_train.append(binvox)
                y_train.append(train_binvox_y)
            else:
                test_binvox_y = check_binvox_y(binvox_id, False)
                if test_binvox_y is not None:
                    x_test.append(binvox)
                    y_test.append(test_binvox_y)
                else:
                    continue

        x_train = np.asarray(x_train, dtype=np.uint8)
        x_test = np.asarray(x_test, dtype=np.uint8)
        y_train = np.asarray(y_train, dtype=np.uint8)
        y_test = np.asarray(y_test, dtype=np.uint8)

        x_train = x_train.reshape(
            [x_train.shape[0], 1] + list(x_train.shape[1:]))
        x_test = x_test.reshape([x_test.shape[0], 1] + list(x_test.shape[1:]))

        if align_data:
            aligned_size = min([len(x_train), len(x_test)])
            x_train = x_train[:aligned_size]
            x_test = x_test[:aligned_size]
            y_train = y_train[:aligned_size]
            y_test = y_test[:aligned_size]

        data = PSBVoxel(x_train, x_test, y_train, y_test)

        if is_cached:
            if is_co_class:
                save_path = PATH_RES_SHAPE_PSB_BINVOX_CACHE_CO_CLASS_ORIGIN
            else:
                save_path = PATH_RES_SHAPE_PSB_BINVOX_CACHE_DEFAULT_ORIGIN
            data.save(save_path)

        return data

    def augment_rotate(self, start, end, step, center, dtype=np.uint8,
                       is_co_class=False, is_cached=False, from_cached=False):
        self._aug(aug_type='rotate', start=start, end=end, step=step,
                  center=center, dtype=dtype, is_co_class=is_co_class,
                  is_cached=is_cached, from_cached=from_cached)

    def augment_translate(self, start, end, step, dtype=np.uint8,
                          is_co_class=False, is_cached=False,
                          from_cached=False):
        self._aug(aug_type='translate', start=start, end=end, step=step,
                  dtype=dtype, is_co_class=is_co_class, is_cached=is_cached,
                  from_cached=from_cached)

    def _aug(self, **kwargs):
        aug_type = kwargs['aug_type']
        is_co_class = kwargs['is_co_class']
        is_cached = kwargs['is_cached']
        from_cached = kwargs['from_cached']
        if aug_type == 'rotate':
            aug_func = super(PSBVoxel, self).augment_rotate
            if is_co_class:
                path = PATH_RES_SHAPE_PSB_BINVOX_CACHE_CO_CLASS_ROTATE
            else:
                path = PATH_RES_SHAPE_PSB_BINVOX_CACHE_DEFAULT_ROTATE
        elif aug_type == 'translate':
            aug_func = super(PSBVoxel, self).augment_translate
            if is_co_class:
                path = PATH_RES_SHAPE_PSB_BINVOX_CACHE_CO_CLASS_TRANS
            else:
                path = PATH_RES_SHAPE_PSB_BINVOX_CACHE_DEFAULT_ROTATE
        else:
            raise TypeError

        if from_cached:
            print 'load augumented voxels from .npy ...'
            try:
                self.set(PSBVoxel.load(path))
                return
            except IOError:
                warnings.warn(
                    'augumented psb_voxel({}) was not loaded.'.format(aug_type))

        # 不要な引数の削除
        del kwargs['aug_type'], \
            kwargs['is_co_class'], \
            kwargs['is_cached'], \
            kwargs['from_cached']

        # Data Augmentation処理
        aug_func(**kwargs)

        if is_cached:
            self.save(path)


class SHRECVoxel(Data3d):
    def __init__(self, x_train, x_test, y_train, y_test):
        super(SHRECVoxel, self).__init__(x_train, x_test, y_train, y_test)

    @staticmethod
    def create_shrec_voxel(n_fold):
        return SHRECVoxel._create(n_fold, parse_binvox,
                                  PATH_RES_SHAPE_SHREC_BINVOX)

    @staticmethod
    def _create(n_fold, parser, file_dir):
        # ID抽出用正規表現
        re_compile = re.compile("\d+")

        cla_items = parse_cla(PATH_RES_SHAPE_SHREC_CLASS_TEST_CLA).items()

        def find_class(binvox_id):
            for i, pair in enumerate(cla_items):
                name, ids = pair
                if binvox_id in ids:
                    return i
            raise IndexError

        x = []
        y = []

        for f in tqdm.tqdm(os.listdir(file_dir)):
            # binvoxファイルフルパス
            fullpath = os.path.join(file_dir, f)
            # binvox配列
            shrec_voxel = parser(fullpath)
            # shrecデータに割り振られたID
            shrec_voxel_id = int(re_compile.findall(f)[0])

            x.append([shrec_voxel, ])
            y.append(find_class(shrec_voxel_id))

        x = np.vstack(x).astype(np.uint8)
        x = x.reshape((x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]))
        y = np.asarray(y, dtype=np.uint8)

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            train_size=1. - 1. / n_fold)

        return SHRECVoxel(x_train, x_test, y_train, y_test)


class SHRECVoxelUSDF(Data3d):
    def __init__(self, x_train, x_test, y_train, y_test):
        super(SHRECVoxelUSDF, self).__init__(x_train, x_test, y_train, y_test)

    @staticmethod
    def create_shrec_voxel_usdf(n_fold):
        return SHRECVoxel._create(n_fold, parse_vxl, PATH_RES_SHAPE_SHREC_VXL)
