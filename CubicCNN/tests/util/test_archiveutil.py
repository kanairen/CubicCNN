#!/usr/bin/env python
# coding: utf-8

import os
import unittest
import zipfile
from CubicCNN.src.util.archiveutil import unzip


class TestArchiveUtil(unittest.TestCase):
    def setUp(self):
        self.path_text = "./testtext.txt"
        self.path_dir = "./test"
        self.path_zip = "{}.zip".format(self.path_dir)
        with open(self.path_text, 'wb') as tf:
            tf.write("test")

        with zipfile.ZipFile(self.path_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(self.path_text)

    def test_unzip(self):
        unzip(self.path_zip, self.path_dir)
        print os.listdir(self.path_dir)
        with open(os.path.join(self.path_dir, self.path_text), 'r') as f:
            for l in f.readlines():
                print l
