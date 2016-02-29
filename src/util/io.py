# coding: utf-8

import os
import numpy as np

__author__ = 'ren'


def numpy_save(f, arr):
    """
    numpy配列を保存
    保存先パスが存在しない場合、新しくディレクトリを作成
    :param f: 保存先パス
    :param arr: numpy配列
    :param allow_pickle: pickle化して保存
    :param fix_imports: python3のオブジェクト配列をpython2に合わせてリネーム
    :return:
    """

    # 保存するファイルのディレクトリ
    f_dir = os.path.dirname(f)
    if not os.path.exists(f_dir):
        os.makedirs(f_dir)
    np.save(f, arr)
