# coding=utf-8

import os
import time
import numpy as np
from src.util.date import ymdt
from src.util.io import numpy_save
from src.helper.visualize import plot_2d
from src.helper.config import *

__author__ = 'Ren'


def learning(model, x_train, x_test, y_train, y_test, n_iter, n_batch,
             show_batch_accuracies=True, save_batch_accuracies=True,
             is_batch_test=False, save_weights_and_biases=True):
    # モデル内訳
    print model

    """
    TRAIN
    """
    # 学習開始時刻文字列(精度保存時に使う)
    ts = ymdt()

    # バッチ数はデータ数の約数
    if len(x_train) % n_batch != 0 or len(x_test) % n_batch != 0:
        raise ValueError("! could not be set n_batch which is not divisor of" +
                         " len(x_train) or len(x_test) !")

    # バッチサイズ
    batch_size_train = len(x_train) / n_batch
    batch_size_test = len(x_test) / n_batch

    # 訓練・テストにおける精度の時系列リスト
    train_accuracies = []
    test_accuracies = []

    for i in range(n_iter):

        print "{}th learning...".format(i)

        start = time.clock()

        sum_train_accuracy = 0
        sum_test_accuracy = 0

        # バッチごとに学習
        for j in range(n_batch):

            print "{}th batch...".format(j)

            from_train = j * batch_size_train
            to_train = (j + 1) * batch_size_train
            from_test = j * batch_size_test if is_batch_test else 0
            to_test = (j + 1) * batch_size_test if is_batch_test else len(
                x_test)

            # 学習時間計算
            train_start = time.clock()
            # 学習
            train_accuracy = model.forward(
                inputs=x_train[from_train:to_train],
                answers=y_train[from_train:to_train],
                is_train=True)
            # 学習時間
            print "train time : ", time.clock() - train_start, "s"

            # テスト時間計測
            test_start = time.clock()
            # テスト
            test_accuracy = model.forward(
                inputs=x_test[from_test:to_test],
                answers=y_test[from_test:to_test],
                is_train=False,
                updates=())
            # テスト時間
            print "test time : ", time.clock() - test_start, "s"

            # 累積精度
            sum_train_accuracy += train_accuracy
            sum_test_accuracy += test_accuracy

            # 平均精度
            ave_train_accuracy = sum_train_accuracy / (j + 1)
            ave_test_accuracy = sum_test_accuracy / (j + 1)

            # バッチテストするかどうかによって出力する精度を変える
            out_test_accuracy = ave_test_accuracy if is_batch_test else test_accuracy

            # バッチごとの精度出力
            if show_batch_accuracies:
                print "train : ", ave_train_accuracy
                print "test : ", out_test_accuracy

            # 精度配列を保存（バッチごと）
            if save_batch_accuracies:
                train_accuracies.append(ave_train_accuracy)
                test_accuracies.append(out_test_accuracy)
                numpy_save(os.path.join(path_res_numpy_array, ts + "_train"),
                           train_accuracies)
                numpy_save(os.path.join(path_res_numpy_array, ts + "_test"),
                           test_accuracies)

        # 一回の学習時間
        print "time : ", time.clock() - start, "s"

        # 精度
        if not show_batch_accuracies:
            print "train : ", sum_train_accuracy / n_batch
            print "test : ", sum_test_accuracy / n_batch

        # 精度配列を保存（データセット一周ごと）
        if not save_batch_accuracies:
            train_accuracies.append(sum_train_accuracy / n_batch)
            test_accuracies.append(sum_test_accuracy / n_batch)
            # 精度の保存（途中で終了しても良いように、一回ごとに更新）
            numpy_save(os.path.join(path_res_numpy_array, ts + "_train"),
                       train_accuracies)
            numpy_save(os.path.join(path_res_numpy_array, ts + "_test"),
                       test_accuracies)

        # 重み・バイアスパラメタの保存
        if save_weights_and_biases:
            model.save_params(ts)

    # グラフの描画
    plot_2d({"train": train_accuracies, "test": test_accuracies},
            x_label="iteration", y_label="accuracy", y_lim=(0, 1))
