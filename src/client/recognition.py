# coding=utf-8

import time
import numpy as np
from src.util.date import ymdt
from src.helper.visualize import plot_2d
from src.helper.config import *

__author__ = 'Ren'


def learning(model, x_train, x_test, y_train, y_test,n_iter,n_batch,
             show_batch_accuracies=True, save_batch_accuracies=True):
    """
    TRAIN
    """
    # 学習開始時刻文字列(精度保存時に使う)
    ts = ymdt()

    # バッチサイズ
    batch_size_train = len(x_train) / n_batch
    batch_size_test = len(x_test) / n_batch

    # 訓練・テストにおける精度の時系列リスト
    train_accuracies = []
    test_accuracies = []

    for i in range(n_iter):

        print "{}th learning...".format(i)

        start = time.clock()

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

            if save_batch_accuracies:
                train_accuracies.append(train_accuracy / (j + 1))
                test_accuracies.append(test_accuracy / (j + 1))
                np.save(path_res_numpy_array + "/" + ts + "_train",
                        train_accuracies)
                np.save(path_res_numpy_array + "/" + ts + "_test",
                        test_accuracies)

        train_accuracy /= n_batch
        test_accuracy /= n_batch

        # 一回の学習時間
        print "time : ", time.clock() - start, "s"

        # 精度
        print "train : ", train_accuracy
        print "test : ", test_accuracy

        if not save_batch_accuracies:
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            # 精度の保存（途中で終了しても良いように、一回ごとに更新）
            np.save(path_res_numpy_array + "/" + ts + "_train",
                    train_accuracies)
            np.save(path_res_numpy_array + "/" + ts + "_test", test_accuracies)

    # グラフの描画
    plot_2d({"train": train_accuracies, "test": test_accuracies},
            x_label="iteration", y_label="accuracy", y_lim=(0, 1))
