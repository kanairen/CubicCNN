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
    # TODO 入力データの可視化
    data = shape.psb_voxel(is_co_class=True, is_cached=True, from_cached=True)
    data.shuffle()
    data.augment_rotate(start=(-5, 0, 0), end=(5, 0, 0),
                        step=(1, 1, 1), center=(50, 50, 50))

    def layer_gen():
        l1 = ConvLayer3d(layer_id=0, shape_size=data.data_shape,
                         activation=calcutil.relu, c_in=1, c_out=16, k=5,
                         s=1, is_dropout=True)
        l2 = MaxPoolLayer3d(layer_id=1, shape_size=l1.output_size,
                            activation=calcutil.identity, c_in=16, k=4)
        l3 = HiddenLayer(layer_id=4, n_in=l2.n_out, n_out=512,
                         activation=calcutil.relu, is_dropout=True)
        l4 = HiddenLayer(layer_id=5, n_in=l3.n_out, n_out=256,
                         activation=calcutil.relu, is_dropout=True)
        l5 = SoftMaxLayer(layer_id=6, n_in=l4.n_out, n_out=len(data.classes()))
        layers = [l1, l2, l3, l4, l5]
        return layers

    model = Model(input_dtype='uint8',layers_gen_func=layer_gen)
    optimizer = Optimizer(data, model)
    optimizer.optimize(100, len(data.x_train) / 100, is_total_test_enabled=False)


if __name__ == '__main__':
    cnn_3d_psb()
