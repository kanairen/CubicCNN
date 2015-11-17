# coding:utf-8

import cProfile
from main import image_recognition

__author__ = 'ren'

if __name__ == '__main__':
    cProfile.run('cubic_cnn()', sort=2)
