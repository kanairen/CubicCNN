# coding:utf-8

import os

import numpy as np

from src.helper.config import path_res_numpy_array
from src.helper.visualize import plot_2d

__author__ = 'ren'


def show_np_array(x_label, y_label, y_lim, location, keyword=None,
                  font_size=15, save=False, seaborn=False):
    arrays = {}
    for f in os.listdir(path_res_numpy_array):
        if keyword is not None and keyword not in f:
            continue
        array = np.load(os.path.join(path_res_numpy_array, f))
        arrays.setdefault(str(f), array)

    plot_2d(arrays, x_label, y_label, locate=location, y_lim=y_lim,
            font_size=font_size, save=save, seaborn=seaborn)


if __name__ == '__main__':
    x_label = "iteration"
    y_label = "accuracy"
    y_lim = (0, 1)
    location = 'upper left'
    keyword = '12-08'
    show_np_array(x_label, y_label, y_lim, location=location,
                  keyword=keyword, save=True)
