# coding:utf-8

from src.helper.decorator import client
from src.helper.psb_helper import PSB
from src.model.mlp.layer import Layer
from src.model.mlp.mlp import MLP

import numpy as np
from src.util.config import path_res_numpy_boxel_test, \
    path_res_numpy_boxel_train

__author__ = 'ren'


# テスト・訓練データのクラス情報の取得 30min fin
# numpyボクセルデータ書き込み 30min fin
# TODO バッチ分割実装 30min/2 TO 11:30
# TODO 繰り返しがくしゅう・テストする機構実装
# TODO 学習→MLP
# TODO グラフプロット 30min

@client
def cubic_cnn(n_div=50):
    """
    DATA
    """
    print "loading data..."
    # 点群データ
    train_inputs, train_answers = PSB.load_boxels(is_test=False)
    test_inputs, test_answers = PSB.load_boxels(is_test=True)

    print train_answers
    print test_answers

    print "train data : ", len(train_inputs)
    print "test data : ", len(test_inputs)
    print "train classes : ", len(set(train_answers))
    print "test classes : ", len(set(test_answers))
    print "total classes : ", len(set(train_answers + test_answers))

    print "saving data..."

    """
    MODEL
    """

    print "preparing models..."

    model = MLP(l1=Layer(n_div ** 3, 2000),
                l2=Layer(2000, 1000),
                l3=Layer(1000, 500))
    model.chain()

    print model.forward((train_inputs[0],))

    """
    TRAIN
    """


if __name__ == '__main__':
    cubic_cnn()
