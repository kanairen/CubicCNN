#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages
import sys

sys.path.append('./CubicCNN')

setup(
    name='CubicCNN',
    version='0.1',
    description='CubicCNN is classifier for 3d shape using convolutional '
                'neural network.',
    packages=find_packages(),
    test_suite='tests.suite'
)
