# coding=utf-8

import numpy as np

__author__ = 'Ren'


def rotate_3d(points, r, rotate_priority=[0, 1, 2]):
    assert len(r) == 3

    rx, ry, rz = r

    points = np.array(points)

    mtr_x = np.array([[1., 0., 0.],
                      [0., np.cos(rx), np.sin(rx)],
                      [0., -np.sin(rx), np.cos(rx)]])
    mtr_y = np.array([[np.cos(ry), 0., -np.sin(ry)],
                      [0., 1., 0.],
                      [np.sin(ry), 0., np.cos(ry)]])
    mtr_z = np.array([[np.cos(rz), np.sin(rz), 0.],
                      [-np.sin(rz), np.cos(rz), 0.],
                      [0., 0., 1.]])

    mtr_a, mtr_b, mtr_c = np.array((mtr_x, mtr_y, mtr_z))[rotate_priority]

    r_points = np.dot(np.dot(np.dot(points, mtr_a), mtr_b), mtr_c)

    return r_points


def trans_3d(points, t):
    t_points = np.copy(points)
    t_points[:, :] += t
    return t_points

