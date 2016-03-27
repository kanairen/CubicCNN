# coding: utf-8

import unittest
import numpy as np
from theano import function, tensor as T
from src.model.layer.conv import ConvLayer2d
from src.model.layer.pool import PoolLayer2d
from src.client.classifier.classifier import learning


class ConvLayer2dTest(unittest.TestCase):
    def setUp(self):
        print ('setUp')

    def test_first(self):
        print ('test first.')

    def test_conv_layer_2d(self):
        n, c, w, h = (1, 1, 5, 5)
        oc = 2
        k = 2

        x = np.ones((n, c, w, h), dtype=np.float32)
        W = np.ones((oc, c, k, k), dtype=np.float32)
        b = np.ones((oc, k, k), dtype=np.float32)

        x_symbol = T.fvectors()

        conv = ConvLayer2d((w, h), c, oc, k, W=W, b=b)
        pool = PoolLayer2d(conv.output_img_size(), oc, k)

        f = function(x_symbol, pool.output(x_symbol))

        print f(x)
