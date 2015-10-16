# coding:utf-8

from src.helper.decorator_helper import client
from src.helper.psb_helper import PSB
from src.model.mlp.layer import Layer
from src.model.mlp.mlp import MLP

import numpy as np
from src.util.config import path_res_numpy_boxel_test, \
    path_res_numpy_boxel_train

__author__ = 'ren'

# TODO numpyボクセルデータ書き込み
# TODO バッチ分割実装

@client
def cubic_cnn(n_div=50):
    """
    DATA
    """
    print "loading data..."
    # PSB
    train_inputs, test_inputs, train_answers, test_answers = PSB.load_vertices_all()

    train_inputs = [PSB.boxel(p, n_div=n_div).flatten() for p in train_inputs]
    test_inputs = [PSB.boxel(p, n_div=n_div).flatten() for p in test_inputs]

    print "saving data..."
    count = 0
    for train, test in zip(train_inputs, test_inputs):
        np.save(path_res_numpy_boxel_test + "/" + str(count), test)
        np.save(path_res_numpy_boxel_train + "/" + str(count), train)
        count += 1
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
