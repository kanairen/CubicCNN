# coding:utf-8

import time
import numpy as np
from src.model.layer.hiddenlayer import HiddenLayer
from src.model.layer.conv import ConvLayer3d
from src.model.layer.pool import PoolLayer3d
from src.model.layerset.mlp import MLP
from src.helper.decorator import client
from src.helper.data import PSB
from src.helper.activation import relu
from src.helper.config import path_res_numpy_array
from src.helper.visualize import plot_2d
from src.util.date import ymdt

__author__ = 'ren'


@client
def solid_recognition(n_div=50, show_batch_accuracies=True):
    x_train, x_test, y_train, y_test, class_labels = PSB.load_boxels(degree=1,
                                                                     is_mixed=True)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    n, n_in = x_train.shape

    print "train data : ", len(x_train)
    print "test data : ", len(x_test)
    print "classes : ", len(set(y_train.tolist() + y_test.tolist()))

    """
    MODEL
    """

    print "preparing models..."

    l1 = ConvLayer3d((50, 50, 50), in_channel=1, out_channel=4, k_size=3,
                     stride=2)
    print l1.output_box_size()
    l2 = PoolLayer3d(l1.output_box_size(), in_channel=4, k_size=3)
    print l2.output_box_size()
    l3 = HiddenLayer(l2.n_out, 500, activation=relu)

    model = MLP(l1=l1, l2=l2, l3=l3, learning_rate=0.01, L1_rate=0.01,
                L2_rate=0.01)

    """
    # TRAIN
    # """
    # 学習開始時刻文字列(精度保存時に使う)
    ts = ymdt()

    # トレーニング繰り返し回数
    n_iter = 100

    # バッチ数
    n_batch = 10

    # バッチサイズ
    batch_size_train = len(x_train) / n_batch
    batch_size_test = len(x_test) / n_batch

    # 訓練・テストにおける精度の時系列リスト
    train_accuracies = []
    test_accuracies = []

    for i in range(n_iter):

        print "{}th learning...".format(i)

        start = time.clock()

        if n_batch == 1:
            train_accuracy = model.forward(x_train, y_train, True)
            test_accuracy = model.forward(x_test, y_test, False, updates=())
        else:
            train_accuracy = 0
            test_accuracy = 0

            # バッチごとに学習
            for j in range(n_batch):
                print "{}th batch...".format(j)

                from_train = j * batch_size_train
                from_test = j * batch_size_test
                to_train = (j + 1) * batch_size_train
                to_test = (j + 1) * batch_size_test

                train_accuracy += model.forward(
                    inputs=x_train[from_train:to_train],
                    answers=y_train[from_train:to_train],
                    is_train=True)
                test_accuracy += model.forward(
                    inputs=x_test[from_test:to_test],
                    answers=y_test[from_test:to_test],
                    is_train=False,
                    updates=())

                if show_batch_accuracies:
                    print "train : ", train_accuracy / (j + 1)
                    print "test : ", test_accuracy / (j + 1)

            train_accuracy /= n_batch
            test_accuracy /= n_batch

        # 一回の学習時間
        print "time : ", time.clock() - start, "s"

        # 精度
        print "train : ", train_accuracy
        print "test : ", test_accuracy

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # 精度の保存（途中で終了しても良いように、一回ごとに更新）
        np.save(path_res_numpy_array + "/" + ts + "_train", train_accuracies)
        np.save(path_res_numpy_array + "/" + ts + "_test", test_accuracies)

    # グラフの描画
    plot_2d({"train": train_accuracies, "test": test_accuracies},
            x_label="iteration", y_label="accuracy", y_lim=(0, 1))


if __name__ == '__main__':
    solid_recognition()
