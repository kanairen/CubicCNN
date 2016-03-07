# coding=utf-8

from src.model.layer.hiddenlayer import HiddenLayer
from src.model.layer.conv import ConvLayer2d
from src.model.layer.pool import PoolLayer2d
from src.model.layerset.mlp import MLP
from src.helper.activation import relu

__author__ = 'Ren'


def cnn(img_size, channel):
    l1 = ConvLayer2d(img_size, channel, 16, k_size=3, activation=relu,
                     is_dropout=True)
    l2 = PoolLayer2d(l1.output_img_size(), 16, k_size=2, activation=lambda x: x)
    l3 = ConvLayer2d(l2.output_img_size(), 16, 32, k_size=3, activation=relu,
                     is_dropout=True)
    l4 = PoolLayer2d(l3.output_img_size(), 32, k_size=2, activation=lambda x: x)
    l5 = HiddenLayer(l4.n_out, 256, is_dropout=True)
    l6 = HiddenLayer(l5.n_out, 10, is_dropout=True, dropout_rate=0.25)

    model = MLP(learning_rate=0.01, L1_rate=0.01,
                l1=l1, l2=l2, l3=l3, l4=l4, l5=l5, l6=l6)

    return model


def cnn_3d_imitation(img_size, channel):
    l1 = ConvLayer2d(img_size, channel, 16, 4, stride=3, activation=relu)
    l2 = PoolLayer2d(l1.output_img_size(), 16, 4)
    l3 = HiddenLayer(l2.n_out, 512, is_dropout=True)
    l4 = HiddenLayer(l3.n_out, 256, is_dropout=True)

    model = MLP(l1=l1, l2=l2, l3=l3, l4=l4, L1_rate=0.0001)
    return model


def for_mnist(img_size):
    l1 = HiddenLayer(img_size[0] * img_size[1], 800, is_dropout=True)
    l2 = HiddenLayer(l1.n_out, 10, is_dropout=True)
    model = MLP(learning_rate=0.01, l1=l1, l2=l2)
    return model
