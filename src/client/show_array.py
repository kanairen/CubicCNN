# coding:utf-8

import os
import numpy as np
from src.util.config import path_res_numpy_array
from src.util.visualize import plot_2d

__author__ = 'ren'


def show_np_array(xlabel, ylabel, ylim, location="lower right", keyword=None,
                  font_size=15, save=False):
    arrays = {}
    for f in os.listdir(path_res_numpy_array):
        if keyword is not None and keyword not in f:
            continue
        array = np.load(path_res_numpy_array + "/" + f)
        arrays.setdefault(str(f), array)

    plot_2d(arrays, xlabel, ylabel, locate=location, y_lim=y_lim,
            font_size=font_size, save=save)


if __name__ == '__main__':
    x_label = "iteration"
    y_label = "accuracy"
    y_lim = (0, 1)
    location = 'upper right'
    keyword = '11-14'
    show_np_array(x_label, y_label, y_lim, location=location, keyword=keyword,
                  save=False)
