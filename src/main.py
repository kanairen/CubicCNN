# coding:utf-8

import numpy as np
from theano import config
from src.helper.psb_helper import PSB
from src.helper.visualize_helper import plot_boxel
from src.helper.decorator_helper import client
from src.model.mlp.mlp import MLP

__author__ = 'ren'


@client
def cubic_cnn():
    """
    DATA
    """
    print "loading data..."
    # PSB
    train_inputs, test_inputs, train_answers, test_answers = PSB.load_vertices_all()

    # train_inputs = PSB.boxel_all(train_inputs)
    # test_inputs = PSB.boxel_all(test_inputs)

    """
    MODEL
    """

    print "preparing models..."

    model = MLP(n_units=[100 * 100 * 100, 1000, 500])

    """
    TRAIN
    """


if __name__ == '__main__':
    cubic_cnn()
