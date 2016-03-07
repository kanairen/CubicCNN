# coding:utf-8

import numpy as np
import classifier
from src.client.structure.image_structure import cnn_3d_imitation
from src.helper.data2d import mnist, cifar10, pattern50_distort, psb_depths
from src.helper.data3d import psb_ids


def image_classification(data_type, n_iter, n_batch,
                         show_batch_accuracies=True,
                         save_batch_accuracies=True,
                         is_batch_test=False):
    # 入力画像の決定
    if data_type == 'mnist':
        x_train, x_test, y_train, y_test = mnist()
    elif data_type == 'cifar':
        x_train, x_test, y_train, y_test = cifar10()
    elif data_type == 'distort':
        x_train, x_test, y_train, y_test = pattern50_distort()
    elif data_type == 'psb_depth':
        train_ids, test_ids = psb_ids('', is_all=True, is_both=True)
        ids = np.r_[(train_ids, test_ids[:len(test_ids) - 1])]
        x_train, x_test, y_train, y_test = psb_depths(ids)

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

    model = cnn_3d_imitation((w, h), c)

    classifier.learning(model, x_train, x_test, y_train, y_test, n_iter,
                        n_batch, show_batch_accuracies, save_batch_accuracies,
                        is_batch_test)
