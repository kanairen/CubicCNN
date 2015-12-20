# coding:utf-8

import os
import itertools
import numpy as np
from collections import OrderedDict
from config import path_res_3d_primitive, path_res_3d_shrec_query, \
    path_res_3d_shrec_target, path_res_3d_psb, path_res_3d_psb_classifier, \
    path_res_3d_psb_binvox
from src.util.parse import parse_obj, parse_cla, parse_off, parse_binvox

__author__ = 'ren'



"""
PRIMITIVE
"""


def primitives(path=path_res_3d_primitive):
    prim_list = []
    for f_name in os.listdir(path):
        prim = parse_obj(os.path.join(path, f_name))
        prim_list.append(prim)
    return prim_list


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


def psb_binvox(id, path=path_res_3d_psb_binvox):
    return parse_binvox(os.path.join(path, "m{}.binvox".format(id)))


def psb_binvoxs(ids, path=path_res_3d_psb_binvox):
    # クラス情報
    path_cls = path_res_3d_psb_classifier
    train_cls = parse_cla(os.path.join(path_cls, "train.cla"))[0]
    test_cls = parse_cla(os.path.join(path_cls, "test.cla"))[0]
    all_cls = reduce(lambda x, y: OrderedDict(x, **y), (train_cls, test_cls))

    # 各データセット別IDリスト
    train_ids = sorted(list(itertools.chain(*train_cls.values())))
    test_ids = sorted(list(itertools.chain(*test_cls.values())))

    # IDから、データのクラスラベルを取得
    # クラスラベルは全クラスリスト中でのIndex
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
            x_train.append(psb_binvox(id, path))
            y_train.append(class_label(train_cls, id))
        elif id in test_ids:
            x_test.append(psb_binvox(id, path))
            y_test.append(class_label(test_cls, id))
        else:
            raise IndexError("psb id:{} is not found!".format(id))

    return x_train, x_test, y_train, y_test


"""
VOXEL
"""


def voxelize(points, n_div=100):
    # -0.5~0.5
    points = __standard(points)
    voxel = np.zeros(shape=(n_div, n_div, n_div), dtype=np.float32)
    for p in points:
        x, y, z = p
        bz = int(z * n_div + n_div) / 2
        by = int(y * n_div + n_div) / 2
        bx = int(x * n_div + n_div) / 2
        voxel[bz][by][bx] = 1

    return voxel


def voxelize_all(points_list, n_div=100):
    return np.array([voxelize(points, n_div) for points in points_list],
                    dtype=np.float32)


def __standard(points):
    mean = np.mean(points, axis=0)
    norm = np.max(points) - np.min(points)
    return np.array([(p - mean) / norm for p in points], dtype=np.float32)
