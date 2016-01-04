# coding=utf-8

import numpy as np
from src.util.sequence import trio

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


def rotate_shapes(shape, r_range, step, rotate_priority=[0, 1, 2]):
    sx, sy, sz = trio(step)
    r_shapes = []
    for rx in xrange(0, r_range[0], sx):
        for ry in xrange(0, r_range[1], sy):
            for rz in xrange(0, r_range[2], sz):
                r_shape = rotate_3d(shape, (rx, ry, rz), rotate_priority)
                r_shapes.append(r_shape)
    return r_shapes


def trans_shapes(shape, t_range, step):
    sx, sy, sz = trio(step)
    t_shapes = []
    for tx in xrange(0, t_range[0], sx):
        for ty in xrange(0, t_range[1], sy):
            for tz in xrange(0, t_range[2], sz):
                t_shape = trans_3d(shape, (tx, ty, tz))
                t_shapes.append(t_shape)
    return t_shapes


def rotate_voxels(voxel, from_r, to_r, step, center, rotate_priority=[0, 1, 2]):
    sx, sy, sz = trio(step)
    return [rotate_voxel(voxel, (rx, ry, rz), center, rotate_priority)
            for rx in xrange(from_r[0], to_r[0], sx)
            for ry in xrange(from_r[1], to_r[1], sy)
            for rz in xrange(from_r[2], to_r[2], sz)]


def trans_voxels(voxel, from_t, to_t, step):
    sx, sy, sz = trio(step)
    return [trans_voxel(voxel, (tx, ty, tz))
            for tx in xrange(from_t[0], to_t[0], sx)
            for ty in xrange(from_t[1], to_t[1], sy)
            for tz in xrange(from_t[2], to_t[2], sz)]


def centerize_voxels(voxels, center):
    assert len(center) == 3
    n, dz, dy, dx = voxels.shape
    c_voxels = np.zeros_like(voxels)
    for i in xrange(n):
        c_voxels[i] = centerize_voxel(voxels[i], center)
    return c_voxels


def rotate_voxel(voxel, r, center, rotate_priority=[0, 1, 2]):
    assert len(r) == 3
    assert len(voxel.shape) == 3

    dz, dy, dx = voxel.shape

    r_x, r_y, r_z = np.asarray(r, dtype=np.float32) / 180. * np.pi

    cx, cy, cz = center

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

    r_voxel = np.zeros_like(voxel)

    for z in xrange(dz):
        for y in xrange(dy):
            for x in xrange(dx):
                if voxel[z][y][x] == 0:
                    continue
                rx, ry, rz = np.dot(
                        np.dot(np.dot((x - cx, y - cy, z - cz), mtr_a), mtr_b),
                        mtr_c)
                if 0 <= rx + cx < dx and 0 <= ry + cy < dy and 0 <= rz + cz < dz:
                    r_voxel[rz + cz][ry + cy][rx + cx] = 1

    return r_voxel


def trans_voxel(voxel, t):
    t_voxel = np.zeros_like(voxel)
    dz, dy, dx = voxel.shape
    tx, ty, tz = t
    for z in xrange(dz):
        for y in xrange(dy):
            for x in xrange(dx):
                if voxel[z][y][x] == 1 and \
                                        0 <= x + tx < dx and \
                                        0 <= y + ty < dy and \
                                        0 <= z + tz < dz:
                    t_voxel[z + tz][y + ty][x + tx] = 1

    return t_voxel


def centerize_voxel(voxel, center):
    assert len(center) == 3
    dz, dy, dx = voxel.shape
    mean_v = np.sum(
            [(x * voxel[z][y][x], y * voxel[z][y][x], z * voxel[z][y][x])
             for z in xrange(dz)
             for y in xrange(dy)
             for x in xrange(dx)], axis=0) / np.sum(voxel)
    c_voxel = trans_voxel(voxel, -(mean_v - center))
    return c_voxel
