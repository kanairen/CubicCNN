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
    __TRAIN_INPUT_NAME = "x_train"
    __TEST_INPUT_NAME = "x_test"
    __TRAIN_ANSWER_NAME = "y_train"
    __TEST_ANSWER_NAME = "y_test"

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.data_shape = x_train.shape[2:]
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)

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

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, self.__TRAIN_INPUT_NAME), self.x_train)
        np.save(os.path.join(path, self.__TEST_INPUT_NAME), self.x_test)
        np.save(os.path.join(path, self.__TRAIN_ANSWER_NAME), self.y_train)
        np.save(os.path.join(path, self.__TEST_ANSWER_NAME), self.y_test)

    @classmethod
    def load(cls, path):
        x_train = np.load(os.path.join(path, cls.__TRAIN_INPUT_NAME) + '.npy')
        x_test = np.load(os.path.join(path, cls.__TEST_INPUT_NAME) + '.npy')
        y_train = np.load(os.path.join(path, cls.__TRAIN_ANSWER_NAME) + '.npy')
        y_test = np.load(os.path.join(path, cls.__TEST_ANSWER_NAME) + '.npy')
        return Data(x_train, x_test, y_train, y_test)

    def set(self, data):
        assert isinstance(data, Data)
        self.x_train = data.x_train
        self.x_test = data.x_test
        self.y_train = data.y_train
        self.y_test = data.y_test

    @staticmethod
    def center(voxel):
        raise NotImplementedError

    def __str__(self):
        return '\n{} : '.format(self.__class__.__name__) + \
               '\nx_train.shape : {}'.format(self.x_train.shape) + \
               '\nx_test.shape : {}'.format(self.x_test.shape) + \
               '\ny_train.shape : {}'.format(self.y_train.shape) + \
               '\ny_test.shape : {}\n'.format(self.y_test.shape)


class Data2d(Data):
    def __init__(self, x_train, x_test, y_train, y_test):
        # shape = (n,c,w,h)
        assert len(x_train.shape) == 4
        assert len(x_test.shape) == 4
        super(Data2d, self).__init__(x_train, x_test, y_train, y_test)

    @classmethod
    def load(cls, path):
        d = super(Data2d, cls).load(path)
        return Data2d(d.x_train, d.x_test, d.y_train, d.y_test)


class Data3d(Data):
    def __init__(self, x_train, x_test, y_train, y_test):
        # shape = (n,c,dx,dy,dz)
        assert len(x_train.shape) == 5
        assert len(x_test.shape) == 5
        super(Data3d, self).__init__(x_train, x_test, y_train, y_test)

    @classmethod
    def load(cls, path):
        d = super(Data3d, cls).load(path)
        return Data3d(d.x_train, d.x_test, d.y_train, d.y_test)

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

    @staticmethod
    def center(voxel):

        # 新しいボクセル
        new_voxel = np.zeros(shape=voxel.shape)

        # 非ゼロインデックス
        non_zeros = np.argwhere(voxel)

        # 平均との差分
        diff = (
            np.average(non_zeros, axis=0) - np.array(voxel.shape) / 2).astype(
            np.int8)

        # 中央寄せした座標
        new_coordinates = non_zeros - diff

        # ボクセルの最大座標値+1
        max_x, max_y, max_z = voxel.shape

        # 新しい座標に対応するボクセルを有効にする
        for x, y, z in new_coordinates:
            # ボクセル空間外の座標はフィルタリング
            if 0 <= x < max_x and 0 <= y < max_y and 0 <= z < max_z:
                new_voxel[x][y][z] = 1
        return new_voxel
