# coding: utf-8
import unittest
from tests.model.layer.hiddenlayer import HiddenLayerTest

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(HiddenLayerTest))
    return suite
