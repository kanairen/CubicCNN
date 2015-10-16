# coding:utf-8

from src.helper.decorator_helper import client
from src.helper.psb_helper import PSB
from src.model.mlp.layer import Layer
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

    model = MLP(l1=Layer(100 * 100 * 100, 2000),
                l2=Layer(2000, 1000),
                l3=Layer(1000, 500))
    model.chain()
    print model.forward(train_inputs[0])

    """
    TRAIN
    """


if __name__ == '__main__':
    cubic_cnn()
