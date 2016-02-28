# coding:utf-8

import os

__author__ = 'ren'

"""
PATH
"""

path = os.path.join(os.path.dirname(__file__), "..", "..")

path_res = os.path.join(path, "res")

# res
path_res_2d = os.path.join(path_res, "2d")
path_res_3d = os.path.join(path_res, "3d")
path_res_numpy = os.path.join(path_res, "numpy")

# res/2d
path_res_2d_pattern = os.path.join(path_res_2d, "pattern")
path_res_2d_psb_depth = os.path.join(path_res_2d, "psb_depth")

# res/3d
path_res_3d_primitive = os.path.join(path_res_3d, "primitive")
path_res_3d_psb = os.path.join(path_res_3d, "psb")
path_res_3d_psb_binvox = os.path.join(path_res_3d, "psb_binvox")
path_res_3d_shrec = os.path.join(path_res_3d, "shrec")

# res/3d/psb
path_res_3d_psb_classifier = os.path.join(path_res_3d_psb, "classifier")

# res/3d/shrec
path_res_3d_shrec_query = os.path.join(path_res_3d_shrec, "query")
path_res_3d_shrec_target = os.path.join(path_res_3d_shrec, "target")

# res/numpy
path_res_numpy_array = os.path.join(path_res_numpy, "array")
path_res_numpy_weights = os.path.join(path_res_numpy, "weights")
path_res_numpy_biases = os.path.join(path_res_numpy, "biases")
path_res_numpy_cache = os.path.join(path_res_numpy, "cache")
path_res_numpy_voxel = os.path.join(path_res_numpy, "voxel")
path_res_numpy_psb = os.path.join(path_res_numpy, "psb")

# res/numpy/voxel
path_res_numpy_voxel_primitive = os.path.join(path_res_numpy_voxel, "primitive")
path_res_numpy_voxel_psb = os.path.join(path_res_numpy_voxel, "psb")

# res/numpy/cache
path_res_numpy_cache_cifar = os.path.join(path_res_numpy_cache, "cache")
path_res_numpy_cache_psb = os.path.join(path_res_numpy_cache, "psb")

# res/numpy/voxel/psb
path_res_numpy_voxel_psb_test = os.path.join(path_res_numpy_voxel_psb, "test")
path_res_numpy_voxel_psb_train = os.path.join(path_res_numpy_voxel_psb, "train")

# res/numpy/psb
path_res_numpy_psb_test = os.path.join(path_res_numpy_psb, "test")
path_res_numpy_psb_train = os.path.join(path_res_numpy_psb, "train")

'''
CUDA
'''
# cuda.get_device(GPU_ID).use()
GPU_ID = 0
