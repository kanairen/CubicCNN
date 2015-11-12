# coding:utf-8

import numpy as np
from src.helper.decorator import client
from src.data.psb import PSB
from src.data.image import Image
from src.helper.visualize import plot_2d
from src.model.layer.hiddenlayer import HiddenLayer
from src.model.layer.conv import ConvLayer2d
from src.model.layer.pool import PoolLayer
from src.model.layerset.mlp import MLP
from src.util.activation import relu
from src.util.config import path_res_2d_pattern, path_res_numpy_array
from src.util.time import ymdt


@client
def cubic_cnn(n_div=50, img_size=(64, 64), is_boxel=False):
    """
    DATA
    """
    print "loading data..."
    if is_boxel:
        # 点群データ
        x_train, y_train = PSB.load_boxels(is_test=False)
        x_test, y_test = PSB.load_boxels(is_test=True)
    else:
        x, y, images, r_images = Image.image_set(path_res_2d_pattern,
                                                 Image.TRANS,
                                                 size=img_size)
        x_train, x_test, y_train, y_test, perm = Image.hold_out(x, y,
                                                                train_rate=0.8)

    # numpy配列にしておくと、スライシングでコピーが発生しない
    x_train = np.array([x.flatten() for x in x_train])
    x_test = np.array([x.flatten() for x in x_test])
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print "train data : ", len(x_train)
    print "test data : ", len(x_test)
    print "classes : ", len(set(y_train.tolist() + y_test.tolist()))

    """
    MODEL
    """

    print "preparing models..."

    n_in = n_div ** 3 if is_boxel else img_size

    l1 = ConvLayer2d(n_in, in_channel=1, out_channel=8, k_size=4,
                     activation=relu, is_dropout=True)
    l2 = PoolLayer(l1.output_img_size(), in_channel=8, k_size=4)
    l3 = HiddenLayer(l2.n_out, 1000)
    l4 = HiddenLayer(l3.n_out, 500)

    model = MLP(l1=l1, l2=l2, l3=l3, l4=l4)
    """
    TRAIN
    """

    # トレーニング繰り返し回数
    n_iter = 1000

    # バッチ数
    n_batch = 1

    # バッチサイズ
    batch_size_train = len(x_train) / n_batch
    batch_size_test = len(x_test) / n_batch

    # 訓練・テストにおける精度の時系列リスト
    train_accuracies = []
    test_accuracies = []

    for i in range(n_iter):

        print "{}st learning...".format(i)

        train_accuracy = 0
        test_accuracy = 0

        if n_batch == 1:
            train_accuracy = model.forward(x_train, y_train, True)
            test_accuracy = model.forward(x_test, y_test, False, updates=())
        else:
            # バッチごとに学習
            for j in range(n_batch):
                print "{}st batch...".format(j)
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
            train_accuracy /= n_batch
            test_accuracy /= n_batch

        print "train : ", train_accuracy
        print "test : ", test_accuracy

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    # グラフの描画
    plot_2d({"train": train_accuracies, "test": test_accuracies},
            xlabel="iteration", ylabel="accuracy", ylim=(0, 1))

    # 精度の保存
    np.save(path_res_numpy_array + "/" + ymdt() + "_train", train_accuracies)
    np.save(path_res_numpy_array + "/" + ymdt() + "_test", test_accuracies)


if __name__ == '__main__':
    cubic_cnn()
