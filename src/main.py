# coding:utf-8

from src.helper.psb_helper import PSB
from src.helper.visualize_helper import plot_boxel
from src.helper.decorator_helper import client

__author__ = 'ren'


@client
def cubic_cnn():
    """
    DATA
    """

    # PSB
    train_inputs, test_inputs, train_answers, test_answers = PSB.load_vertices_all()

    train_inputs = PSB.boxel_all(train_inputs)
    test_inputs = PSB.boxel_all(test_inputs)

    plot_boxel(train_inputs[0])

    """
    MODEL
    """

    # model = MLP(128 * 128, 1000, 500)

    """
    TRAIN
    """

if __name__ == '__main__':
    cubic_cnn()
