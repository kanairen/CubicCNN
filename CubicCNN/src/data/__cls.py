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

    def load(self, url, archive_name):
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
