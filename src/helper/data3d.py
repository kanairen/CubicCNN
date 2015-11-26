# coding:utf-8

import os
import numpy as np
from numpy import sin, cos
from sklearn.cross_validation import train_test_split
from config import path_res_3d_primitive
from src.util.parse import parse_obj

__author__ = 'ren'


def primitive_rotate(rotate_priority=[0, 1, 2], to=(5, 5, 4), step=(1, 1, 1),
                     test_size=0.2):
    primitives = __primitive()
    r_primitives = []
    for prim in primitives:
        r_primitives.extend(__rotate_3d(prim, rotate_priority, to, step))
    ids = [i / len(primitives) for i in xrange(len(r_primitives))]
    return train_test_split(r_primitives, ids, test_size=test_size)


def primitive_trans(to=(5, 5, 4), step=(1, 1, 1), test_size=0.2):
    primitives = __primitive()
    t_primitives = []
    for prim in primitives:
        t_primitives.extend(__trans_3d(prim, to, step))
    ids = [i / len(primitives) for i in xrange(len(t_primitives))]
    return train_test_split(t_primitives, ids, test_size=test_size)


def __primitive(path=path_res_3d_primitive):
    primitives = []
    for f_name in os.listdir(path):
        prim = parse_obj(path + "/" + f_name)
        primitives.append(prim)
    return primitives


def __rotate_3d(points, rotate_priority=[0, 1, 2], to=(5, 5, 4),
                step=(1, 1, 1)):
    points = np.array(points)

    to_x, to_y, to_z = to
    step_x, step_y, step_z = step
    ranges = (xrange(0, to_x, step_x),
              xrange(0, to_y, step_y),
              xrange(0, to_z, step_z))

    mtr_x = lambda r: np.array([[1., 0., 0.],
                                [0., cos(r), sin(r)],
                                [0., -sin(r), cos(r)]])
    mtr_y = lambda r: np.array([[cos(r), 0., -sin(r)],
                                [0., 0., 0.],
                                [sin(r), 0., cos(r)]])
    mtr_z = lambda r: np.array([[cos(r), sin(r), 0.],
                                [-sin(r), cos(r), 0.],
                                [0., 0., 1.]])

    a_range, b_range, c_range = np.array(ranges)[rotate_priority]
    mtr_a, mtr_b, mtr_c = np.array((mtr_x, mtr_y, mtr_z))[rotate_priority]

    rotate_points = []
    for da in a_range:
        for db in b_range:
            for dc in c_range:
                r_points = np.dot(np.dot(np.dot(points, mtr_a(da)), mtr_b(db)),
                                  mtr_c(dc))
                rotate_points.append(r_points)

    return rotate_points


def __trans_3d(points, to=(5, 5, 4), step=(1, 1, 1)):
    trans_points = []
    to_x, to_y, to_z = to
    step_x, step_y, step_z = step
    for p in points:
        for dx in xrange(0, to_x, step_x):
            for dy in xrange(0, to_y, step_y):
                for dz in xrange(0, to_z, step_z):
                    trans_p = (p[0] + dx, p[1] + dy, p[2] + dz)
                    trans_points.append(trans_p)

    return trans_points
