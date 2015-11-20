# coding:utf-8

import PIL.Image
import numpy as np
import sequence

__author__ = 'ren'


def gradation_8bit(img):
    if isinstance(img, PIL.Image.Image):
        img = np.asarray(img)

    img -= img.min()
    img /= img.max()
    img *= 255
    img = img.astype(np.uint8)

    return img


def translate(img, tx, ty):
    return img.transform(size=img.size,
                         method=PIL.Image.AFFINE,
                         data=(1, 0, tx, 0, 1, ty))


def translate_with_zoom(image, pw, ph, zoom=1.5):
    w, h = image.size
    cropped_image = image.crop(
        (pw, ph, pw + int(w / zoom), ph + int(h / zoom)))
    resize_image = cropped_image.resize((w, h))

    return resize_image


def scale(image, pw, ph):
    w, h = image.size
    cropped_image = image.crop((pw, ph, w - pw, h - ph))
    resize_image = cropped_image.resize((w, h))

    return resize_image


def distort(img, newsize, rnd=None):
    assert isinstance(img, PIL.Image.Image)

    # 圧縮前後のサイズ
    w, h = img.size
    new_w, new_h = sequence.pair(newsize)

    # 圧縮時の画素選択範囲サイズ
    kw = w / new_w
    kh = h / new_h

    new_array = np.zeros(newsize)

    if rnd is None:
        rnd = np.random.RandomState()

    for j in xrange(new_h):
        for i in xrange(new_w):
            old_i = rnd.randint(low=0, high=kw) + i * kw
            old_j = rnd.randint(low=0, high=kh) + j * kh
            new_array[j][i] = img.getpixel((old_j, old_i))

    distorted_image = PIL.Image.fromarray(new_array)

    return distorted_image
