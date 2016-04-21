#!/usr/bin/env python
# coding: utf-8

import os
import urllib
import itertools
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

    def augment_rotate(self, from_angles, to_angles, step_angles, center,
                       rotate_priority=[0, 1, 2], dtype=np.uint8):
        # NOTE 境界値は範囲に含む
        fx, fy, fz = from_angles
        tx, ty, tz = to_angles
        sx, sy, sz = step_angles

        n_inc = ((tx - fx) / sx + 1) * ((ty - fy) / sy + 1) * (
            (tz - fz) / sz + 1)

        def augment(x_data):
            new_data = np.empty(
            for i in xrange(len(x_data)):
                [x_data.shape[0] * n_inc] + list(x_data.shape[1:]), dtype=dtype)
                x = x_data[i, 0]
                idx = 0
                for ax in xrange(fx, tx + 1, sx):
                    for ay in xrange(fy, ty + 1, sy):
                        for az in xrange(fz, tz + 1, sz):
                            new_data[n_inc * i + idx] = \
                                self._rotate(x, (ax, ay, az),
                                             center,
                                             rotate_priority)
                            idx += 1
            return new_data

        self.x_train = augment(self.x_train)
        self.x_test = augment(self.x_test)
        self.y_train = np.asarray(
            list(itertools.chain(*[[y] * n_inc for y in self.y_train])))
        self.y_test = np.asarray(
            list(itertools.chain(*[[y] * n_inc for y in self.y_test])))

    @staticmethod
    def _rotate(voxel, angle, center, rotate_priority=[0, 1, 2],
                dtype=np.uint8):

        # 弧度
        r_x, r_y, r_z = np.asarray(angle, dtype=np.float32) / 180. * np.pi

        # 回転行列
        mtr_x = np.array([[1., 0., 0.],
                          [0., np.cos(r_x), np.sin(r_x)],
                          [0., -np.sin(r_x), np.cos(r_x)]])
        mtr_y = np.array([[np.cos(r_y), 0., -np.sin(r_y)],
                          [0., 1., 0.],
                          [np.sin(r_y), 0., np.cos(r_y)]])
        mtr_z = np.array([[np.cos(r_z), np.sin(r_z), 0.],
                          [-np.sin(r_z), np.cos(r_z), 0.],
                          [0., 0., 1.]])

        m1, m2, m3 = np.array((mtr_x, mtr_y, mtr_z))[rotate_priority]

        r_voxel = np.zeros_like(voxel, dtype=dtype)

        new_xyz = np.dot(np.dot(np.dot(np.argwhere(voxel) - center, m1), m2),
                         m3) + center

        dx, dy, dz = voxel.shape

        for ix, iy, iz in new_xyz.astype(np.uint8):
            if 0 <= ix < dx and 0 <= iy < dy and 0 <= iz < dz:
                r_voxel[ix][iy][iz] = 1

        return r_voxel
