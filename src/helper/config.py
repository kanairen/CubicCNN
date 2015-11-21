# coding:utf-8

import os

__author__ = 'ren'

"""
PATH
"""

path = os.path.dirname(__file__) + "/../.."

path_res = path + "/res"

# res
path_res_2d = path_res + "/2d"
path_res_3d = path_res + "/3d"
path_res_numpy = path_res + "/numpy"

# res/2d
path_res_2d_pattern = path_res_2d + "/pattern"

# res/3d
path_res_3d_psb = path_res_3d + "/psb"
path_res_3d_shrec = path_res_3d + "/shrec"

# res/3d/psb
path_res_3d_psb_classifier = path_res_3d_psb + "/classifier"

# res/3d/shrec
path_res_3d_shrec_query = path_res_3d_shrec + "/query"
path_res_3d_shrec_target = path_res_3d_shrec + "/target"

# res/numpy
path_res_numpy_psb = path_res_numpy + "/psb"
path_res_numpy_boxel = path_res_numpy + "/boxel"
path_res_numpy_array = path_res_numpy + "/array"

# res/numpy/psb
path_res_numpy_psb_test = path_res_numpy_psb + "/test"
path_res_numpy_psb_train = path_res_numpy_psb + "/train"

# res/numpy/boxel
path_res_numpy_boxel_test = path_res_numpy_boxel + "/test"
path_res_numpy_boxel_train = path_res_numpy_boxel + "/train"

'''
CUDA
'''
# cuda.get_device(GPU_ID).use()
GPU_ID = 0
