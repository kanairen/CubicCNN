# coding=utf-8

import numpy as np

__author__ = 'Ren'


def rotate_3d(points, rotate_priority=[0, 1, 2], to=(5, 5, 4),
              step=(1, 1, 1)):
    points = np.array(points)

    to_x, to_y, to_z = to
    step_x, step_y, step_z = step
    ranges = (xrange(0, to_x, step_x),
              xrange(0, to_y, step_y),
              xrange(0, to_z, step_z))

    mtr_x = lambda r: np.array([[1., 0., 0.],
                                [0., np.cos(r), np.sin(r)],
                                [0., -np.sin(r), np.cos(r)]])
    mtr_y = lambda r: np.array([[np.cos(r), 0., -np.sin(r)],
                                [0., 1., 0.],
                                [np.sin(r), 0., np.cos(r)]])
    mtr_z = lambda r: np.array([[np.cos(r), np.sin(r), 0.],
                                [-np.sin(r), np.cos(r), 0.],
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
    print str(f.readline()).rstrip().split(' ')


def trans_3d(points, to=(5, 5, 4), step=(1, 1, 1)):
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
