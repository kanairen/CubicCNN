# coding:utf-8

import time
import numpy as np
from src.helper.activation import relu
from src.helper.config import path_res_numpy_array
from src.helper.data import mnist, cifar10, pattern50_distort
from src.helper.decorator import client
from src.helper.visualize import plot_2d, merge_images
from src.model.layer.conv import ConvLayer2d
from src.model.layer.hiddenlayer import HiddenLayer
from src.model.layer.pool import PoolLayer2d
from src.model.layerset.mlp import MLP
from src.util.date import ymdt


@client
def image_recognition(n_div=50, type='mnist', show_batch_accuracies=True,
                      save_batch_accuracies=True):
    # 入力画像の決定
    if type == 'distort':
        x_train, x_test, y_train, y_test = pattern50_distort()
    elif type == 'cifar':
        x_train, x_test, y_train, y_test = cifar10()
    elif type == 'mnist':
        x_train, x_test, y_train, y_test = mnist()

    # 入力画像のチャネルとサイズ
    n, c, h, w = x_train.shape

    # 入力データの平坦化
    x_train = x_train.reshape(len(x_train), c * w * h)
    x_test = x_test.reshape(len(x_test), c * w * h)

    print "train data : ", len(x_train)
    print "test data : ", len(x_test)
    print "classes : ", len(set(y_train.tolist() + y_test.tolist()))

    """
    MODEL
    """

    print "preparing models..."

    l1 = ConvLayer2d((h, w), c, 16, k_size=3, activation=relu)
    l2 = PoolLayer2d(l1.output_img_size(), 16, k_size=2, activation=lambda x: x)
    l3 = ConvLayer2d(l2.output_img_size(), 16, 32, k_size=3, activation=relu)
    l4 = PoolLayer2d(l3.output_img_size(), 32, k_size=2, activation=lambda x: x)
    l5 = ConvLayer2d(l4.output_img_size(), 32, 32, k_size=3, activation=relu)
    l6 = PoolLayer2d(l5.output_img_size(), 32, k_size=2, activation=lambda x: x)
    l7 = HiddenLayer(l6.n_out, 10)

    model = MLP(learning_rate=0.01, L1_rate=0.0001, L2_rate=0.001,
                l1=l1, l2=l2, l3=l3, l4=l4, l5=l5, l6=l6, l7=l7)
    """
    TRAIN
    """
    # 学習開始時刻文字列(精度保存時に使う)
    ts = ymdt()

    # トレーニング繰り返し回数
    n_iter = 100

    # バッチ数
    n_batch = 100

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

    # 畳み込み層のフィルタ画像を保存
    merge_images(l1.filter_image(), (w, h), pad=10).save(ymdt() + ".png")

    # グラフの描画
    plot_2d({"train": train_accuracies, "test": test_accuracies},
            x_label="iteration", y_label="accuracy", y_lim=(0, 1))

