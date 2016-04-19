#!/usr/bin/env python
# coding: utf-8

import unittest
import zipfile
from CubicCNN.src.util.archiveutil import unzip


class TestArchiveUtil(unittest.TestCase):
    def setUp(self):
        print "TestArchiveUtil : setUp"
        with open("./testtext.txt", 'wb') as tf:
            tf.write("test")

        with zipfile.ZipFile("./test.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write("./testtext.txt")

    def test_unzip(self):
        print "TestArchiveUtil : test_unzip"
        unzip("./test.zip", "./test")
        with open("./test/testtext.txt", 'r') as f:
            for l in f.readlines():
                print l

