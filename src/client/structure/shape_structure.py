# coding=utf-8

from src.model.layer.hiddenlayer import HiddenLayer
from src.model.layer.conv import ConvLayer3d
from src.model.layer.pool import PoolLayer3d
from src.model.layerset.mlp import MLP
from src.helper.activation import relu

__author__ = 'Ren'


def cnn(box_size):
    l1 = ConvLayer3d(box_size, 1, 16, 4, stride=3, activation=relu)
    l2 = PoolLayer3d(l1.output_box_size(), 16, 4)
    l3 = HiddenLayer(l2.n_out, 512, is_dropout=True)
    l4 = HiddenLayer(l3.n_out, 256, is_dropout=True)

    model = MLP(l1=l1, l2=l2, l3=l3, l4=l4, L1_rate=0.0001)

    return model
