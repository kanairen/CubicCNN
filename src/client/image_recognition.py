# coding:utf-8

import recognition
from src.model.layer.hiddenlayer import HiddenLayer
from src.model.layer.conv import ConvLayer2d
from src.model.layer.pool import PoolLayer2d
from src.model.layerset.mlp import MLP
from src.helper.data import mnist, cifar10, pattern50_distort
from src.helper.activation import relu


def image_recognition(data_type, n_iter, n_batch, show_batch_accuracies=True,
                      save_batch_accuracies=True):
    # 入力画像の決定
    if data_type == 'distort':
        x_train, x_test, y_train, y_test = pattern50_distort()
    elif data_type == 'cifar':
        x_train, x_test, y_train, y_test = cifar10()
    elif data_type == 'mnist':
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

    recognition.learning(model, x_train, x_test, y_train, y_test, n_iter,
                         n_batch, show_batch_accuracies, save_batch_accuracies)
