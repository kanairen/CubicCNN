# coding:utf-8

import numpy as np
from src.helper.decorator import client
from src.helper.psb import PSB
from src.helper.visualize import plot_2d
from src.helper.image import Image
from src.model.mlp.layer import Layer
from src.model.mlp.mlp import MLP
from src.util.config import path_res_2d_pattern

__author__ = 'ren'


# TODO バッチ分割実装 15min
# TODO テストデータシャッフル 15min(18:35)
# TODO 誤りデータの表示と誤り例（深層学習　p）1h
# TODO Dropoutの実装 1h

# TODO CNN
# TODO フィルタ実装
# TODO プレーンフィルタ実装
# TODO スポークフィルタ実装


@client
def cubic_cnn(n_div=50, is_boxel=False):
    """
    DATA
    """
    print "loading data..."
    if is_boxel:
        # 点群データ
        train_inputs, train_answers = PSB.load_boxels(is_test=False)
        test_inputs, test_answers = PSB.load_boxels(is_test=True)
    else:
        inputs, answers, images, r_images = Image.image_set(path_res_2d_pattern,
                                                            Image.TRANS)
        train_inputs, test_inputs, train_answers, test_answers, perm = Image.hold_out(
            inputs, answers, train_rate=0.8)

    train_inputs = [inputs.flatten() for inputs in train_inputs]
    test_inputs = [inputs.flatten() for inputs in test_inputs]

    print len(train_answers)
    print len(test_answers)

    print "train data : ", len(train_inputs)
    print "test data : ", len(test_inputs)
    print "train classes : ", len(set(train_answers))
    print "test classes : ", len(set(test_answers))

    """
    MODEL
    """

    print "preparing models..."

    n_in = n_div ** 3 if is_boxel else 128 * 128

    model = MLP(l1=Layer(n_in, 2000),
                l2=Layer(2000, 1000),
                l3=Layer(1000, 500))
    model.chain()

    """
    TRAIN
    """
    train_accuracies = []
    test_accuracies = []
    for i in range(10000):
        print "{}st learning...".format(i)
        train_accuracy = model.forward(inputs=train_inputs,
                                       answers=train_answers)
        test_accuracy = model.forward(inputs=test_inputs,
                                      answers=test_answers,
                                      updates=())
        print "train : ", train_accuracy
        print "test : ", test_accuracy
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    plot_2d(xlabel="iteration", ylabel="accuracy", train=train_accuracies,
            test=test_accuracies)


if __name__ == '__main__':
    cubic_cnn()
