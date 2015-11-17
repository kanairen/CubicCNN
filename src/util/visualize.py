# coding:utf-8

import six
import numpy as np
import PIL.Image
import date
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


def plot_2d(array_dict, x_label, y_label, x_lim=None, y_lim=None, grid=True,
            locate="lower right", font_size=20, seaborn=False, save=False):
    if seaborn:
        import seaborn

    # グラフの描画
    for name, value in six.iteritems(array_dict):
        pyplot.plot(value, label=name)
    pyplot.legend(loc=locate)
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)

    # 表示範囲
    if x_lim:
        pyplot.xlim(x_lim)
    if y_lim:
        pyplot.ylim(y_lim)

    # グリッド
    if grid:
        if seaborn:
            seaborn.set_style('whitegrid')
        else:
            pyplot.grid()

    # フォントサイズ
    if seaborn:
        seaborn.set(font_scale=font_size)
    else:
        pyplot.rcParams['font.size'] = font_size

    if save:
        pyplot.savefig(date.ymdt() + ".png")

    pyplot.show()


def merge_images(images, img_size, row=1, pad=0, bgcolor=0):
    # 各画像のサイズ
    w, h = img_size

    # 大枠のサイズ
    cw = (w + pad) * (len(images) / row) + pad
    ch = (h + pad) * row + pad

    # 複数画像をまとめた画像
    canvas = PIL.Image.new(images[0].mode, (cw, ch), bgcolor)

    # 複数の画像を一枚の画像に敷き詰めるように貼り付ける
    x = pad
    y = pad
    for image in images:
        image = image.resize(img_size)
        canvas.paste(image, box=(x, y))
        x += w + pad
        if x >= cw:
            x = pad
            y += h + pad

    return canvas
