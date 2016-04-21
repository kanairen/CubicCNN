# coding: utf-8


import os

PATH = os.path.dirname(__file__)

PATH_RES = os.path.join(PATH, 'res')

# /res
PATH_RES_IMAGE = os.path.join(PATH_RES, 'image')
PATH_RES_SHAPE = os.path.join(PATH_RES, 'shape')
PATH_RES_TMP = os.path.join(PATH_RES, 'tmp')

# /res/shape
PATH_RES_SHAPE_PSB = os.path.join(PATH_RES_SHAPE, 'psb')

# /res/shape/psb
PATH_RES_SHAPE_PSB_OFF = os.path.join(PATH_RES_SHAPE_PSB, 'off')
PATH_RES_SHAPE_PSB_CLASS = os.path.join(PATH_RES_SHAPE_PSB, 'class')
PATH_RES_SHAPE_PSB_BINVOX = os.path.join(PATH_RES_SHAPE_PSB, 'binvox')

# /res/shape/psb/binvox
PATH_RES_SHAPE_PSB_BINVOX_CACHE = os.path.join(PATH_RES_SHAPE_PSB_BINVOX,
                                               'cache')

# /res/shape/psb/binvox/cache
PATH_RES_SHAPE_PSB_BINVOX_CACHE_DEFAULT = \
    os.path.join(PATH_RES_SHAPE_PSB_BINVOX_CACHE, 'default')
PATH_RES_SHAPE_PSB_BINVOX_CACHE_CO_CLASS = \
    os.path.join(PATH_RES_SHAPE_PSB_BINVOX_CACHE, 'co_class')

"""
PSB
"""

PSB_DATASET_URL = 'http://shape.cs.princeton.edu/benchmark/download.cgi?file=download/psb_v1.zip'

PSB_RELATIVE_CLA_PATH = "benchmark/classification/v1/base"
PSB_RELATIVE_OFF_PATH = "benchmark/db"

PSB_CLS_TEST = "test.cla"
PSB_CLS_TRAIN = "train.cla"

