# coding:utf-8

import numpy as np

__author__ = 'ren'




def img_col(img, kh, kw, sy, sx, ph, pw, pval=0, cover_all=False):
    n, c, h, w = img.shape
    out_h = conv_outsize(h, kh, sy, ph, cover_all)
    out_w = conv_outsize(w, kw, sx, pw, cover_all)

    img = np.pad(img,
                 ((0, 0), (0, 0), (ph, ph + sy - 1), (pw, pw + sx - 1)),
                 mode='constant', constant_values=(pval,))
    col = np.ndarray((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    for i in range(kh):
        i_lim = i + sy * out_h
        for j in range(kw):
            j_lim = j + sx * out_w
            col[:, :, i, j, :, :] = img[:, :, i:i_lim:sy, j:j_lim:sx]

    return col


def conv_outsize(size, k, s, p, cover_all=False):
    if cover_all:
        return (size + p * 2 - k + s - 1) // s + 1
    else:
        return (size + p * 2 - k) // s + 1
