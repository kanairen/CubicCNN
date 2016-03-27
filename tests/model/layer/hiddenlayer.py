# coding: utf-8

import unittest
from src.model.layer.hiddenlayer import HiddenLayer


class HiddenLayerTest(unittest.TestCase):
    def setUp(self):
        print ('setUp')

    def test_first(self):
        print ('test first.')

    def test_hidden_layer(self):
        n_in = 100
        n_out = 10

        layer = HiddenLayer(n_in, n_out)





