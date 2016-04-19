#!/usr/bin/env python
# coding: utf-8


import data, model, util, test_main

import unittest
import inspect

test_modules = [data, model, util, test_main]


def suite():
    suite = unittest.TestSuite()
    for m in test_modules:
        for label, cls in inspect.getmembers(m, predicate=inspect.isclass):
            if isinstance(cls, unittest.TestCase):
                suite.addTest(unittest.makeSuite(cls))
                print "Add : {}".format(label)
    return suite
