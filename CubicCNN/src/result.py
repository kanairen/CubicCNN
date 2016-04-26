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
        self.time_stamp = datetime.datetime.today().strftime(
            '%Y-%m-%d_%H:%M:%S')

    def add(self, key, value):
        array = self.results.setdefault(key, np.asarray([]))
        self.results[key] = np.append(array, value)

    def add_all(self, items):
        for k, v in items:
            self.add(k, v)

    def set(self, key, array):
        self.results[key] = np.asarray(array)

    def save(self):
        for key, array in self.results.items():
            if not os.path.exists(
                    os.path.join(PATH_RES_RESULT, self.time_stamp)):
                os.makedirs(os.path.join(PATH_RES_RESULT, self.time_stamp))
            np.save(os.path.join(PATH_RES_RESULT, self.time_stamp, key), array)

    @staticmethod
    def load(time_stamp):
        result = Result()
        for f in os.listdir(os.path.join(PATH_RES_RESULT, time_stamp)):
            array = np.load(os.path.join(PATH_RES_RESULT, time_stamp, f))
            result.set(f.replace('.npy', ''), array)
        return result

    def sub_result(self, keys):
        new_result = Result()
        for key in keys:
            new_result.set(key, self.results[key])
        return new_result
