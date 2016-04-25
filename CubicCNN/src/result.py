#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import datetime
from collections import OrderedDict

from CubicCNN import PATH_RES_RESULT


class Result(object):
    def __init__(self):
        self.results = OrderedDict()

    def add(self, key, value):
        array = self.results.setdefault(key, [])
        array.append(value)

    def add_all(self, items):
        for k, v in items:
            self.add(k, v)

    def set(self, key, array):
        self.results[key] = array

    def save(self):
        for key, array in self.results.items():
            name = key + datetime.datetime.today().strftime(
                '_%Y-%m-%d_%H:%M:%S')
            np.save(os.path.join(PATH_RES_RESULT, name), array)

    @staticmethod
    def load(file_name, key_name=None):
        result = Result()
        file_name = os.path.splitext(file_name)[0] + ".npy"
        array = np.load(os.path.join(PATH_RES_RESULT, file_name))
        if key_name is None:
            key_name = file_name.replace(
                '_\d{4}-\d{1,2}-\d{1,2}_\d{2}:\d{2}:\d{2}', '').replace('.npy',
                                                                        '')
        result.add(key_name, array)
        return result

    def sub_result(self, keys):
        new_result = Result()
        for key in keys:
            new_result.set(key, self.results[key])
        return new_result
