# coding:utf-8

import PIL.Image
import numpy as np

__author__ = 'ren'


def gradation_8bit(img):
    if type(img) == PIL.Image:
        img = np.asarray(img)

    img -= img.min()
    img /= img.max()
    img *= 255
    img = img.astype(np.uint8)

    return img
