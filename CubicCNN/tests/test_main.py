# coding: utf-8

import unittest
from CubicCNN.src.data import image
from CubicCNN.src.model.model import Model
from CubicCNN.src.model.layer.__hidden import HiddenLayer
from CubicCNN.src.model.layer.__conv import ConvLayer2d
from CubicCNN.src.model.layer.__pool import MaxPoolLayer2d
from CubicCNN.src.model.layer.__softmax import SoftMaxLayer
from CubicCNN.src.optimizer import Optimizer
from CubicCNN.src.util import calcutil


class TestCnn2d(unittest.TestCase):
    def setUp(self):
        print "setUp"

        def layer_gen():
            l1 = ConvLayer2d(layer_id=0, image_size=d.data_shape,
                             activation=calcutil.relu, c_in=1, c_out=16,
                             k=(2, 2),
                             s=(1, 1))
            l2 = MaxPoolLayer2d(layer_id=1, image_size=l1.output_image_size,
                                activation=calcutil.identity, c_in=16, k=(2, 2))
            l3 = ConvLayer2d(layer_id=2, image_size=l2.output_image_size,
                             activation=calcutil.relu, c_in=16, c_out=32,
                             k=(2, 2),
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

        d = image.mnist()
        m = Model(layer_gen)
        self.optimizer = Optimizer(d, m)

    def test_cnn_2d(self):
        print "test_cnn_2d"
        self.optimizer.optimize(3, 10)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCnn2d))
    return suite
