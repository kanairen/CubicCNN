# coding:utf-8

import os
import itertools
import numpy as np
from collections import OrderedDict
from sklearn.cross_validation import train_test_split
from config import path_res_3d_primitive, path_res_3d_shrec_query, \
    path_res_3d_shrec_target, path_res_3d_psb, path_res_3d_psb_classifier
from src.util.parse import parse_obj, parse_cla, parse_off
from src.util.solid import rotate_3d, trans_3d

__author__ = 'ren'

"""
PRIMITIVE
"""


def primitive_rotate(rotate_priority=[0, 1, 2], to=(5, 5, 4), step=(1, 1, 1),
                     test_size=0.2):
    primitives = __primitive()
    r_primitives = []
    for prim in primitives:
        r_primitives.extend(rotate_3d(prim, rotate_priority, to, step))
    ids = [i / len(primitives) for i in xrange(len(r_primitives))]
    return train_test_split(r_primitives, ids, test_size=test_size)


def primitive_trans(to=(5, 5, 4), step=(1, 1, 1), test_size=0.2):
    primitives = __primitive()
    t_primitives = []
    for prim in primitives:
        t_primitives.extend(trans_3d(prim, to, step))
    ids = [i / len(primitives) for i in xrange(len(t_primitives))]
    return train_test_split(t_primitives, ids, test_size=test_size)


def __primitive(path=path_res_3d_primitive):
    primitives = []
    for f_name in os.listdir(path):
        prim = parse_obj(path + "/" + f_name)
        primitives.append(prim)
    return primitives


"""
SHREC
"""


def shrec():
    target_list = os.listdir(path_res_3d_shrec_target)
    query_list = os.listdir(path_res_3d_shrec_query)
    targets = []
    queries = []
    for f_target, f_query in zip(target_list, query_list):
        target_vertices = parse_off(path_res_3d_shrec_target + f_target)[0]
        query_vertices = parse_off(path_res_3d_shrec_query + f_query)[1]
        targets.append(target_vertices)
        queries.append(query_vertices)
    return targets, queries


"""
PSB
"""


def psb(file_id, path=path_res_3d_psb):
    vertices, faces = parse_off(os.path.join(path, "m{}.off".format(file_id)))
    return vertices


def psbs(ids, path=path_res_3d_psb):
    path_cls = path_res_3d_psb_classifier
    train_cls, train_tree = parse_cla(os.path.join(path_cls, "train.cla"))
    test_cls, test_tree = parse_cla(os.path.join(path_cls, "test.cla"))
    all_cls = reduce(lambda x, y: OrderedDict(x, **y), (train_cls, test_cls))
    train_ids = sorted(list(itertools.chain(*train_cls.values())))
    test_ids = sorted(list(itertools.chain(*test_cls.values())))

    def class_label(cls, id):
        for cls_name, cls_ids in cls.items():
            if id in cls_ids:
                return all_cls.keys().index(cls_name)
        raise IndexError("psb id:{} is not found!".format(id))

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for id in ids:
        if id in train_ids:
            x_train.append(psb(id, path))
            y_train.append(class_label(train_cls, id))
        elif id in test_ids:
            x_test.append(psb(id, path))
            y_test.append(class_label(test_cls, id))
        else:
            raise IndexError("psb id:{} is not found!".format(id))

    return x_train, x_test, y_train, y_test

"""
BOXEL
"""


def boxel(points, n_div=100):
    # -0.5~0.5
    points = __standard(points)
    boxel = np.zeros(shape=(n_div, n_div, n_div), dtype=np.float32)
    for p in points:
        x, y, z = p
        bz = int(z * n_div + n_div) / 2
        by = int(y * n_div + n_div) / 2
        bx = int(x * n_div + n_div) / 2
        boxel[bz][by][bx] = 1

    return boxel


def boxel_all(points_list, n_div=100):
    return np.array([boxel(points, n_div) for points in points_list],
                    dtype=np.float32)


def __standard(points):
    mean = np.mean(points, axis=0)
    norm = np.max(points) - np.min(points)
    return np.array([(p - mean) / norm for p in points], dtype=np.float32)
