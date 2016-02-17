# coding: utf-8

import unittest
from src.helper import data3d


class Data3dTest(unittest.TestCase):
    def setUp(self):
        print('setUp')

    def test_first(self):
        print('test_first')

    def test_perspective_projection(self):
        points = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        width, height = (100, 100)

        def assert_pp():
            pp = data3d.perspective_projection(points, width, height)
            answer = [-1 for i in xrange(width * height)]
            for p in points:
                answer[width * p[1] + p[0]] = p[3] / 100.
            return pp == answer

        self.assertTrue(assert_pp())


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(Data3dTest))
    return test_suite
