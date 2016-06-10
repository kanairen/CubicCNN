# coding: utf-8


import os

PATH = os.path.dirname(__file__)

PATH_RES = os.path.join(PATH, 'res')

# /res
PATH_RES_IMAGE = os.path.join(PATH_RES, 'image')
PATH_RES_RESULT = os.path.join(PATH_RES, 'result')
PATH_RES_SHAPE = os.path.join(PATH_RES, 'shape')
PATH_RES_TMP = os.path.join(PATH_RES, 'tmp')

# /res/shape
PATH_RES_SHAPE_PSB = os.path.join(PATH_RES_SHAPE, 'psb')
PATH_RES_SHAPE_SHREC = os.path.join(PATH_RES_SHAPE, 'shrec')

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

# /res/shape/psb/binvox/cache/default
PATH_RES_SHAPE_PSB_BINVOX_CACHE_DEFAULT_ORIGIN = \
    os.path.join(PATH_RES_SHAPE_PSB_BINVOX_CACHE_DEFAULT, 'origin')
PATH_RES_SHAPE_PSB_BINVOX_CACHE_DEFAULT_ROTATE = \
    os.path.join(PATH_RES_SHAPE_PSB_BINVOX_CACHE_DEFAULT, 'rotate')
PATH_RES_SHAPE_PSB_BINVOX_CACHE_DEFAULT_TRANS = \
    os.path.join(PATH_RES_SHAPE_PSB_BINVOX_CACHE_DEFAULT, 'trans')

# /res/shape/psb/binvox/cache/co_class
PATH_RES_SHAPE_PSB_BINVOX_CACHE_CO_CLASS_ORIGIN = \
    os.path.join(PATH_RES_SHAPE_PSB_BINVOX_CACHE_CO_CLASS, 'origin')
PATH_RES_SHAPE_PSB_BINVOX_CACHE_CO_CLASS_ROTATE = \
    os.path.join(PATH_RES_SHAPE_PSB_BINVOX_CACHE_CO_CLASS, 'rotate')
PATH_RES_SHAPE_PSB_BINVOX_CACHE_CO_CLASS_TRANS = \
    os.path.join(PATH_RES_SHAPE_PSB_BINVOX_CACHE_CO_CLASS, 'trans')

# /res/shape/psb/class
PATH_RES_SHAPE_SHREC_CLASS = os.path.join(PATH_RES_SHAPE_SHREC, 'class')
PATH_RES_SHAPE_SHREC_BINVOX = os.path.join(PATH_RES_SHAPE_SHREC, 'binvox')

# /res/shape/psb/class
PATH_RES_SHAPE_SHREC_CLASS_TEST_CLA = os.path.join(PATH_RES_SHAPE_SHREC_CLASS,
                                                   'test.cla')

"""
PSB
"""

PSB_DATASET_URL = 'http://shape.cs.princeton.edu/benchmark/download.cgi?file=download/psb_v1.zip'

PSB_RELATIVE_CLA_PATH = "benchmark/classification/v1/base"
PSB_RELATIVE_OFF_PATH = "benchmark/db"

PSB_CLS_TEST = "test.cla"
PSB_CLS_TRAIN = "train.cla"
