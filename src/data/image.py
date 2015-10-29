# coding:utf-8

import os
import numpy as np
import itertools
from PIL.Image import AFFINE, Image, open

__author__ = 'ren'


class Image(object):
    # 画像変換のパラメタ
    DEFAULT_TRANSLATE_STEP = 2
    DEFAULT_TRANS_RANGE_X = (-4, 4)
    DEFAULT_TRANS_RANGE_Y = (-5, 5)

    DEFAULT_ROTATE_STEP = 1
    DEFAULT_ROTATE_ANGLE = 20
    # 画像変換指定用ラベル
    TRANS = "TRANSLATE"
    ROTATE = "ROTATE"

    @classmethod
    def image_set(cls, target_path, process_type, size=(128, 128),
                  is_flatten=False):
        # 学習器入力リスト
        inputs = []
        # 正解データリスト
        answers = []
        # 元画像リスト
        images = []
        # 変換後画像リスト
        r_images = []

        for i, f_name in enumerate(os.listdir(target_path)):
            # 元画像
            image = open(target_path + "/" + f_name).convert('1').resize(
                size=size)
            images.append(image)

            # 指定された変換を画像に適用し、変換結果をリストで受け取る
            if process_type == cls.TRANS:
                processed_imgs = Image.trans_with_range(image)
            elif process_type == cls.ROTATE:
                processed_imgs = Image.rotate_with_range(image)
            else:
                processed_imgs = [image]
            r_images.extend(processed_imgs)

            # 画像の画素値配列を一次元に
            arrays = [np.asarray(np.uint8(img)).flatten() if is_flatten
                      else np.asarray(np.uint8(img)) for img in processed_imgs]

            # 入力リストに画像データを追加
            inputs.extend(arrays)
            # 正解リストに正解ラベルを追加
            answers.extend([i] * len(arrays))

        return inputs, answers, images, r_images

    @staticmethod
    def hold_out(inputs, answers, train_rate):
        # 入力・正解データを訓練用・テスト用に分割
        perm = np.random.permutation(len(inputs))
        n_train = int(len(inputs) * train_rate)
        train_in, test_in = np.split(
            np.array([inputs[i] for i in perm], dtype=np.float32), [n_train])
        train_ans, test_ans = np.split(
            np.array([answers[i] for i in perm], dtype=np.int32), [n_train])

        return train_in, test_in, train_ans, test_ans, perm

    @staticmethod
    def scale(image, pw, ph):

        w, h = image.size
        cropped_image = image.crop((pw, ph, w - pw, h - ph))
        resize_image = cropped_image.resize((w, h))

        return resize_image

    @staticmethod
    def translate_with_zoom(image, pw, ph, zoom=1.5):

        w, h = image.size
        cropped_image = image.crop(
            (pw, ph, pw + int(w / zoom), ph + int(h / zoom)))
        resize_image = cropped_image.resize((w, h))

        return resize_image

    @staticmethod
    def translate(image, tx, ty):
        return image.transform(size=image.size,
                               method=AFFINE,
                               data=(1, 0, tx, 0, 1, ty))

    @classmethod
    def trans_with_range(cls, img, range_x=DEFAULT_TRANS_RANGE_X,
                         range_y=DEFAULT_TRANS_RANGE_Y,
                         step=DEFAULT_TRANSLATE_STEP):
        assert len(range_x) == 2
        assert len(range_y) == 2
        # 画像の平行移動
        t_range_x = range(range_x[0], range_x[1], step)
        t_range_y = range(range_y[0], range_y[1], step)
        t_imgs_2d = [[cls.translate(img, w, h) for w in t_range_x] for h in
                     t_range_y]
        return list(itertools.chain(*t_imgs_2d))

    @classmethod
    def rotate_with_range(cls, img, angle=DEFAULT_ROTATE_ANGLE,
                          step=DEFAULT_ROTATE_STEP):
        return [img.rotate(r) for r in range(angle)]
