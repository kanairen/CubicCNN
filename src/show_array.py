# coding:utf-8

import os

import numpy as np

from src.util.config import path_res_numpy_array
from src.util.visualize import plot_2d

__author__ = 'ren'


def show_np_array(keyword=None):
    arrays = {}
    for f in os.listdir(path_res_numpy_array):
        if keyword is not None and keyword not in f:
            continue
        array = np.load(path_res_numpy_array + "/" + f)
        arrays.setdefault(str(f), array)

    xlabel = "iteration"
    ylabel = "accuracy"
    y_lim = (0, 1)
    location = "upper right"
    plot_2d(arrays, xlabel, ylabel, locate=location, ylim=y_lim, font_size=15)


if __name__ == '__main__':
    show_np_array('-05')
