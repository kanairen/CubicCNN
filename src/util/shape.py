# coding=utf-8

import numpy as np

__author__ = 'Ren'


def rotate_3d(points, r, rotate_priority=[0, 1, 2]):
    assert len(r) == 3

    points = np.array(points)

    r_x, r_y, r_z = np.asarray(r, dtype=np.float32) / 180. * np.pi

    mtr_x = np.array([[1., 0., 0.],
                      [0., np.cos(r_x), np.sin(r_x)],
                      [0., -np.sin(r_x), np.cos(r_x)]])
    mtr_y = np.array([[np.cos(r_y), 0., -np.sin(r_y)],
                      [0., 1., 0.],
                      [np.sin(r_y), 0., np.cos(r_y)]])
    mtr_z = np.array([[np.cos(r_z), np.sin(r_z), 0.],
                      [-np.sin(r_z), np.cos(r_z), 0.],
                      [0., 0., 1.]])

    mtr_a, mtr_b, mtr_c = np.array((mtr_x, mtr_y, mtr_z))[rotate_priority]

    r_points = np.dot(np.dot(np.dot(points, mtr_a), mtr_b), mtr_c)

    return r_points


def trans_3d(points, t):
    t_points = np.copy(points)
    t_points[:, :] += t
    return t_points

