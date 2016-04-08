# coding: utf-8

from model.layer.__hidden import HiddenLayer
from model.layer.__conv import ConvLayer2d
from model.layer.__pool import MaxPoolLayer2d
from model.layer.__softmax import SoftMaxLayer
from data import image
from model.model import Model
from optimizer import Optimizer

from util import calcutil


def main():
    d = image.mnist()

    def layer_gen():
        l1 = ConvLayer2d(layer_id=0, image_size=d.data_shape,
                         activation=calcutil.relu, c_in=1, c_out=16, k=(2, 2),
                         s=(1, 1))
        l2 = MaxPoolLayer2d(layer_id=1, image_size=l1.output_image_size,
                            activation=calcutil.identity, c_in=16, k=(2, 2))
        l3 = ConvLayer2d(layer_id=2, image_size=l2.output_image_size,
                         activation=calcutil.relu, c_in=16, c_out=32, k=(2, 2),
                         s=(1, 1))
        l4 = MaxPoolLayer2d(layer_id=3, image_size=l3.output_image_size,
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
    optimizer.optimize(10,100)


if __name__ == '__main__':
    main()
