#!/usr/bin/env python
# coding: utf-8

import os
import urllib
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

    def rotate(self, angle, center, rotate_priority=(0, 1, 2)):
        # メモリ溢れ防止のため、１つずつ書き換え
        # NOTE ガーベジコレクションによるオーバヘッドの可能性
        for i in xrange(len(self.x_train)):
            self.x_train[i] = self._rotate(self.x_train[i], angle, center,
                                           rotate_priority=rotate_priority)
        for j in xrange(len(self.x_test)):
            self.x_test[j] = self._rotate(self.x_test, angle, center,
                                          rotate_priority=rotate_priority)

    @staticmethod
    def _rotate(voxel, angle, center, rotate_priority=(0, 1, 2)):

        dz, dy, dx = voxel.shape

        r_x, r_y, r_z = np.asarray(angle, dtype=np.float32) / 180. * np.pi

        cx, cy, cz = center

        mtr_x = np.array([[1., 0., 0.],
                          [0., np.cos(r_x), np.sin(r_x)],
                          [0., -np.sin(r_x), np.cos(r_x)]])
        mtr_y = np.array([[np.cos(r_y), 0., -np.sin(r_y)],
                          [0., 1., 0.],
                          [np.sin(r_y), 0., np.cos(r_y)]])
        mtr_z = np.array([[np.cos(r_z), np.sin(r_z), 0.],
                          [-np.sin(r_z), np.cos(r_z), 0.],
                          [0., 0., 1.]])

        first, second, third = np.array((mtr_x, mtr_y, mtr_z))[rotate_priority]

        r_voxel = np.zeros_like(voxel)

        for z in xrange(dz):
            for y in xrange(dy):
                for x in xrange(dx):
                    if voxel[z][y][x] == 0:
                        continue
                    rx, ry, rz = np.dot(
                        np.dot(np.dot((x - cx, y - cy, z - cz), first), second),
                        third)
                    if 0 <= rx + cx < dx and 0 <= ry + cy < dy and 0 <= rz + cz < dz:
                        r_voxel[rz + cz][ry + cy][rx + cx] = 1

        return r_voxel
