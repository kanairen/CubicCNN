# coding:utf-8

import numpy as np
import recognizer
from src.model.layer.hiddenlayer import HiddenLayer
from src.model.layerset.mlp import MLP
from src.helper.decorator import client
from src.helper.data3d import load_psb_boxels, primitive_rotate, \
    primitive_trans, boxel_all

__author__ = 'ren'


@client
def solid_recognition(n_iter, n_batch, show_batch_accuracies=False,
                      save_batch_accuracies=False, **kwargs):
    data_type = kwargs['data_type']

    if data_type == 'psb':
        x_train, x_test, y_train, y_test, class_labels = load_psb_boxels(4)
    elif data_type == 'primitive_rotate':
        x_train, x_test, y_train, y_test = primitive_rotate()
        x_train = boxel_all(x_train)
        x_test = boxel_all(x_test)
    elif data_type == 'primitive_trans':
        x_train, x_test, y_train, y_test = primitive_trans()
        x_train = boxel_all(x_train)
        x_test = boxel_all(x_test)

    box_size = (100, 100, 100)
    n_in = reduce(lambda x, y: x * y, box_size)

    x_train = np.array(x_train).reshape(len(x_train), n_in)
    x_test = np.array(x_test).reshape(len(x_test), n_in)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print "train data : ", len(x_train)
    print "test data : ", len(x_test)
    print "classes : ", len(set(y_train.tolist() + y_test.tolist()))

    """
    MODEL
    """

    print "preparing models..."

    # l1 = ConvLayer3d(box_size, 1, 16, 4, stride=3, activation=relu)
    # l2 = PoolLayer3d(l1.output_box_size(), 16, 4)
    l1 = HiddenLayer(n_in, 512, is_dropout=True)
    l2 = HiddenLayer(l1.n_out, 512, is_dropout=True)

    model = MLP(l1=l1, l2=l2, L1_rate=0.0001)

    """
    # TRAIN
    # """

    recognizer.learning(model, x_train, x_test, y_train, y_test, n_iter,
                         n_batch, show_batch_accuracies, save_batch_accuracies)
