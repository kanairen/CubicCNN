# coding: utf-8


import os

PATH = os.path.dirname(__file__)

PATH_RES = os.path.join(PATH, 'res')

PATH_RES_IMAGE = os.path.join(PATH_RES, 'image')
PATH_RES_SHAPE = os.path.join(PATH_RES, 'shape')
PATH_RES_TMP = os.path.join(PATH_RES, 'tmp')

PATH_RES_SHAPE_PSB = os.path.join(PATH_RES_SHAPE, 'psb')

PATH_RES_SHAPE_PSB_OFF = os.path.join(PATH_RES_SHAPE_PSB, 'off')
PATH_RES_SHAPE_PSB_CLASS = os.path.join(PATH_RES_SHAPE_PSB, 'class')
PATH_RES_SHAPE_PSB_BINVOX = os.path.join(PATH_RES_SHAPE_PSB, 'binvox')

"""
PSB
"""

PSB_DATASET_URL = 'http://shape.cs.princeton.edu/benchmark/download.cgi?file=download/psb_v1.zip'

PSB_RELATIVE_CLA_PATH = "benchmark/classification/v1/base"
PSB_RELATIVE_OFF_PATH = "benchmark/db"

PSB_CLS_TEST = "test.cla"
PSB_CLS_TRAIN = "train.cla"
