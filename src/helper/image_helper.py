# coding:utf-8

import itertools

__author__ = 'ren'


class ImageHelper(object):
    # 画像変換のパラメタ
    TRANSLATE_STEP = 2
    MAX_TRANSLATE_RANGE_X = 8
    MAX_TRANSLATE_RANGE_Y = 10
    MAX_ANGLE = 20

    @classmethod
    def trans(cls, img):
        # 画像の平行移動
        t_range_x = range(-cls.MAX_TRANSLATE_RANGE_X / 2,
                          cls.MAX_TRANSLATE_RANGE_X / 2,
                          cls.TRANSLATE_STEP)
        t_range_y = range(-cls.MAX_TRANSLATE_RANGE_Y / 2,
                          cls.MAX_TRANSLATE_RANGE_Y / 2,
                          cls.TRANSLATE_STEP)
        t_imgs_2d = [[cls.translate(img, w, h) for w in t_range_x] for h in
                     t_range_y]
        return list(itertools.chain(*t_imgs_2d))

    @classmethod
    def rotate(cls, img):
        return [img.rotate(r) for r in range(cls.MAX_ANGLE)]
