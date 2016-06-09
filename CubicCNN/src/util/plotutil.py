#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


def plot_voxel(voxels, color='blue', facecolor='white', alpha=1.0, x_label="X",
               y_label="Y", z_label="Z"):
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

    # 描画ボクセル
    px, py, pz = np.where(np.asarray(voxels))

    # 表示範囲を設定
    lim_min = min((px.min(), py.min(), pz.min()))
    lim_max = max((px.max(), py.max(), pz.max()))
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_zlim(lim_min, lim_max)

    # 背景色
    ax.patch.set_facecolor(facecolor)

    # 背景透明度
    ax.patch.set_alpha(alpha)

    ax.plot3D(px, py, pz, "s", color=color)

    pyplot.show()
