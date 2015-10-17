# coding:utf-8

import six
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

__author__ = 'ren'


def plot_3d(points,
            x_label="X",
            y_label="Y",
            z_label="Z",
            x_lim=(-1, 1),
            y_lim=(-1, 1),
            z_lim=(-1, 1)):
    """
    点群を３次元グラフ上にプロット
    :param points:
    :return:
    """

    # グラフの作成
    fig = pyplot.figure()
    ax = Axes3D(fig)

    # 軸ラベルを設定
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    # 表示範囲を設定
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_zlim(z_lim[0], z_lim[1])

    # 描画
    ax.plot(np.array([p[1] for p in points], dtype=float),
            np.array([p[0] for p in points], dtype=float),
            np.array([p[2] for p in points], dtype=float),
            "o",
            ms=1,
            mew=0.5)
    pyplot.show()


def plot_boxel(boxels,
               x_label="X",
               y_label="Y",
               z_label="Z"):
    """
    点群を３次元グラフ上にプロット
    :param points:
    :return:
    """

    # グラフの作成
    fig = pyplot.figure()
    ax = Axes3D(fig)

    # 軸ラベルを設定
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    # 描画
    rx, ry, rz = boxels.shape
    px, py, pz = [], [], []
    for z in range(rz):
        for y in range(ry):
            for x in range(rx):
                if boxels[z][y][x] == 1:
                    px.append(x)
                    py.append(y)
                    pz.append(z)
    # 表示範囲を設定
    lim_min, lim_max = min(px + py + pz), max(px + py + pz)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_zlim(lim_min, lim_max)

    ax.plot3D(np.array(px, dtype=int),
              np.array(py, dtype=int),
              np.array(pz, dtype=int), "s")
    pyplot.show()


def plot_2d(xlabel, ylabel, xlim=None, ylim=None, **args):
    # グラフの描画
    for name, values in six.iteritems(args):
        pyplot.plot(values, label=name)
    pyplot.legend()
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    if xlim:
        pyplot.xlim(xlim)
    if ylim:
        pyplot.ylim(ylim)
    pyplot.show()
