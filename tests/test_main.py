# coding: utf-8
import unittest
from tests.model.layer.hiddenlayer import HiddenLayerTest
from tests.model.layer.conv import ConvLayer2dTest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(HiddenLayerTest))
    suite.addTest(unittest.makeSuite(ConvLayer2dTest))
    return suite
