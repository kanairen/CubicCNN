# coding:utf-8

import os
import numpy as np
from numpy import sin, cos
from config import path_res_3d_primitive
from src.util.parse import parse_obj

__author__ = 'ren'


def primitive(path=path_res_3d_primitive):
    primitives = []
    for f_name in os.listdir(path):
        prim = parse_obj(path + "/" + f_name)
        primitives.append(prim)
    return primitives


