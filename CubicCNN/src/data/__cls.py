#!/usr/bin/env python
# coding: utf-8

import os
import urllib
import itertools
import tqdm
import numpy as np
from CubicCNN.src.util.archiveutil import unzip


class DataLoader(object):
    def __init__(self, data_home, archive_home, root_name):
        self.data_home = data_home
        self.archive_home = archive_home
        self.root_name = root_name

    def _load(self, url, archive_name):
        ext = os.path.splitext(url.split('/')[-1])[1]
        from_path = os.path.join(self.archive_home, archive_name + ext)
        to_path = os.path.join(self.archive_home, archive_name)

        if os.path.exists(to_path):
            return
        elif os.path.exists(from_path):
            self.__deploy(ext, from_path, to_path)
        else:
            if not os.path.exists(self.archive_home):
                os.makedirs(self.archive_home)
            urllib.urlretrieve(url, filename=from_path)
            self.__deploy(ext, from_path, to_path)

    @staticmethod
    def __deploy(ext, from_path, to_path):
        if ext == ".zip":
            unzip(from_path, to_path)
        else:
            raise NotImplementedError


class Data(object):
    def __init__(self, x_train, x_test, y_train, y_test, data_shape):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.data_shape = data_shape

    def data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test

    def shuffle(self):
        perm_train = np.random.permutation(len(self.x_train))
        perm_test = np.random.permutation(len(self.x_test))
        self.x_train = self.x_train[perm_train]
        self.y_train = self.y_train[perm_train]
        self.x_test = self.x_test[perm_test]
        self.y_test = self.y_test[perm_test]

    def classes(self):
        return list(set(self.y_train))


class Data2d(Data):
    def __init__(self, x_train, x_test, y_train, y_test):
        # shape = (n,c,w,h)
        assert len(x_train.shape) == 4
        assert len(x_test.shape) == 4
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        data_shape = x_train.shape[2:]
        super(Data2d, self).__init__(x_train, x_test, y_train, y_test,
                                     data_shape)


class Data3d(Data):
    def __init__(self, x_train, x_test, y_train, y_test):
        # shape = (n,c,dx,dy,dz)
        assert len(x_train.shape) == 5
        assert len(x_test.shape) == 5
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        data_shape = x_train.shape[2:]
        super(Data3d, self).__init__(x_train, x_test, y_train, y_test,
                                     data_shape)

    def augment_rotate(self, start, end, step, center, dtype=np.uint8):
        def rotate(voxel, angle):
            return self._rotate(voxel, angle, center, dtype)

        self._augment(start, end, step, rotate, dtype=dtype)

    def _augment(self, start, end, step, func, dtype=np.uint8):
        # NOTE 境界値は範囲に含む
        start_x, start_y, start_z = start
        end_x, end_y, end_z = end
        step_x, step_y, step_z = step

        n_inc = ((end_x - start_x) / step_x + 1) * \
                ((end_y - start_y) / step_y + 1) * \
                ((end_z - start_z) / step_z + 1)

        def augment(x_data):
            new_data = np.empty(
                [x_data.shape[0] * n_inc] + list(x_data.shape[1:]), dtype=dtype)
            for i in tqdm.tqdm(xrange(len(x_data))):
                x = x_data[i, 0]
                idx = 0
                for ax in xrange(start_x, end_x + 1, step_x):
                    for ay in xrange(start_y, end_y + 1, step_y):
                        for az in xrange(start_z, end_z + 1, step_z):
                            new_data[n_inc * i + idx] = func(x, (ax, ay, az))
                            idx += 1
            return new_data

        self.x_train = augment(self.x_train)
        self.x_test = augment(self.x_test)
        self.y_train = np.asarray(
            list(itertools.chain(*[[y] * n_inc for y in self.y_train])))
        self.y_test = np.asarray(
            list(itertools.chain(*[[y] * n_inc for y in self.y_test])))

    def augment_translate(self, start, end, step, dtype=np.uint8):
        self._augment(start, end, step, self._translate, dtype=dtype)

    @staticmethod
    def _translate(voxel, trans, dtype=np.uint8):
        t_voxel = np.zeros_like(voxel, dtype=dtype)
        dx, dy, dz = voxel.shape
        for ix, iy, iz in np.argwhere(voxel) + trans:
            if 0 <= ix < dx and 0 <= iy < dy and 0 <= iz < dz:
                t_voxel[int(ix)][int(iy)][int(iz)] = 1
        return t_voxel

    @staticmethod
    def _rotate(voxel, angle, center, dtype=np.uint8):
        # TODO すべてのxyz軸順序を網羅した行列を用意(現在はxyzのみ)

        # 弧度
        rx, ry, rz = np.asarray(angle, dtype=np.float32) / 180. * np.pi

        # 回転行列(x→y→z)
        mtr = np.array(
            [[np.cos(rx) * np.cos(ry) * np.cos(rz) - np.sin(rx) * np.sin(rz),
              -np.cos(rx) * np.cos(ry) * np.sin(rz) - np.sin(rx) * np.cos(rz),
              np.cos(rx) * np.sin(ry)],
             [np.sin(rx) * np.cos(ry) * np.cos(rz) + np.cos(rx) * np.sin(rz),
              -np.sin(rx) * np.cos(ry) * np.sin(rz) + np.cos(rx) * np.cos(rz),
              np.sin(rx) * np.sin(ry)],
             [-np.sin(ry) * np.cos(rz), np.sin(ry) * np.sin(rz), np.cos(ry)]])

        new_xyz = np.dot(np.argwhere(voxel) - center, mtr) + center

        dx, dy, dz = voxel.shape

        r_voxel = np.zeros_like(voxel, dtype=dtype)
        for ix, iy, iz in new_xyz:
            if 0 <= ix < dx and 0 <= iy < dy and 0 <= iz < dz:
                r_voxel[int(ix)][int(iy)][int(iz)] = 1

        return r_voxel
