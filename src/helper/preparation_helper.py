# coding:utf-8

import os
import numpy as np
from PIL import Image

__author__ = 'ren'


class PreparationHelper(object):
    # 画像変換指定用ラベル
    TRANS = "TRANSLATE"
    ROTATE = "ROTATE"

    @classmethod
    def images_and_answers(cls, target_path, process_type, is_flatten=False):
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
            image = Image.open(target_path + "/" + f_name).convert('1')
            images.append(image)

            # 指定された変換を画像に適用し、変換結果をリストで受け取る
            if process_type == cls.TRANS:
                cls.trans = cls.trans(image)
                processed_imgs = cls.trans
            elif process_type == cls.ROTATE:
                processed_imgs = cls.rotate(image)
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
