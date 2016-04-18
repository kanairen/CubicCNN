#!/usr/bin/env python
# coding: utf-8

from model.layer.__hidden import HiddenLayer
from model.layer.__conv import ConvLayer2d, ConvLayer3d
from model.layer.__pool import MaxPoolLayer2d, MaxPoolLayer3d
from model.layer.__softmax import SoftMaxLayer
from data import image
from data import shape
from model.model import Model
from optimizer import Optimizer

from util import calcutil


def cnn_2d_mnist():
    d = image.mnist()
    d.shuffle()

    def layer_gen():
        l1 = ConvLayer2d(layer_id=0, image_size=d.data_shape,
                         activation=calcutil.relu, c_in=1, c_out=16, k=(2, 2),
                         s=(1, 1), is_dropout=True)
        l2 = MaxPoolLayer2d(layer_id=1, image_size=l1.output_size,
                            activation=calcutil.identity, c_in=16, k=(2, 2))
        l3 = ConvLayer2d(layer_id=2, image_size=l2.output_size,
                         activation=calcutil.relu, c_in=16, c_out=32, k=(2, 2),
                         s=(1, 1), is_dropout=True)
        l4 = MaxPoolLayer2d(layer_id=3, image_size=l3.output_size,
                            activation=calcutil.identity, c_in=32, k=(2, 2))
        l5 = HiddenLayer(layer_id=4, n_in=l4.n_out, n_out=800,
                         activation=calcutil.relu)
        l6 = HiddenLayer(layer_id=5, n_in=l5.n_out, n_out=100,
                         activation=calcutil.relu)
        l7 = SoftMaxLayer(layer_id=6, n_in=l6.n_out, n_out=len(d.classes()))
        layers = [l1, l2, l3, l4, l5, l6, l7]
        return layers

    m = Model(layer_gen)
    optimizer = Optimizer(d, m)
    optimizer.optimize(100, 1000)


def cnn_3d_psb():
    data = shape.psb_voxel()
    data.shuffle()

    def layer_gen():
        l1 = ConvLayer3d(layer_id=0, shape_size=data.data_shape,
                         activation=calcutil.relu, c_in=1, c_out=16, k=2,
                         s=1, is_dropout=True)
        l2 = MaxPoolLayer3d(layer_id=1, shape_size=l1.output_size,
                            activation=calcutil.identity, c_in=16, k=2)
        l3 = ConvLayer3d(layer_id=2, shape_size=l2.output_size,
                         activation=calcutil.relu, c_in=16, c_out=32, k=2,
                         s=1, is_dropout=True)
        l4 = MaxPoolLayer3d(layer_id=3, shape_size=l3.output_size,
                            activation=calcutil.identity, c_in=32, k=2)
        l5 = HiddenLayer(layer_id=4, n_in=l4.n_out, n_out=800,
                         activation=calcutil.relu)
        l6 = HiddenLayer(layer_id=5, n_in=l5.n_out, n_out=100,
                         activation=calcutil.relu)
        l7 = SoftMaxLayer(layer_id=6, n_in=l6.n_out, n_out=len(data.classes()))
        layers = [l1, l2, l3, l4, l5, l6, l7]
        return layers

    model = Model(layer_gen)
    optimizer = Optimizer(data, model)
    optimizer.optimize(100, 100)


if __name__ == '__main__':
    import numpy as np

    ary = np.ones((1, 2, 3, 4, 5))
    label = ['n', 'c', 'x', 'y', 'z']
    ary = np.rollaxis(ary, 4, 1)
    ary = np.rollaxis(ary, 4, 1)
    ary = np.rollaxis(ary, 4, 1)
    ary = np.rollaxis(ary, 4, 1)
    print [label[i - 1] for i in ary.shape]
    exit()
    cnn_3d_psb()
