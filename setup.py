# coding: utf-8

from setuptools import setup, find_packages
import sys

sys.path.append('./CubicCNN/src')
sys.path.append('./CubicCNN/tests')

setup(
    name='CubicCNN',
    version='0.1',
    description='CubicCNN is classifier for 3d shape using convolutional '
                'neural network.',
    packages=find_packages(),
    test_suite='test_main.suite'
)
