# coding:utf-8

import recognition
from src.client.structure.image_structure import cnn
from src.helper.data import mnist, cifar10, pattern50_distort


def image_recognition(data_type, n_iter, n_batch,
                      show_batch_accuracies=True, save_batch_accuracies=True):
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

    model = cnn((w, h), c)

    recognition.learning(model, x_train, x_test, y_train, y_test, n_iter,
                         n_batch, show_batch_accuracies, save_batch_accuracies)
