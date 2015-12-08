# coding:utf-8

import os
import itertools
import numpy as np
from collections import OrderedDict
from numpy import sin, cos
from sklearn.cross_validation import train_test_split
from config import path_res_3d_primitive, path_res_3d_psb, \
    path_res_3d_psb_classifier, path_res_numpy_psb_test, \
    path_res_numpy_psb_train, path_res_3d_shrec_target, path_res_3d_shrec_query, \
    path_res_numpy_boxel_psb_test, path_res_numpy_boxel_psb_train
from src.util.parse import parse_obj, parse_off, parse_cla

__author__ = 'ren'

"""
PRIMITIVE
"""


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
                                [0., 1., 0.],
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
    print str(f.readline()).rstrip().split(' ')


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


### Deprecated ###

# def load_psb_vertices_all(cls, degree, dir=path_res_3d_psb):
#     train_label_info, test_label_info, class_labels = cls.__label_info(
#         degree)
#
#     train_vertices = []
#     test_vertices = []
#     train_answers = []
#     test_answers = []
#
#     for label, ids in train_label_info.items():
#         for id in ids:
#             vertices = cls.__load_vertices("{}.off".format(id), dir, False)
#             train_vertices.append(vertices)
#         train_answers += [class_labels.index(label)] * len(ids)
#
#     for label, ids in test_label_info.items():
#         for id in ids:
#             vertices = cls.__load_vertices("{}.off".format(id), dir, True)
#             test_vertices.append(vertices)
#         test_answers += [class_labels.index(label)] * len(ids)
#
#     return train_vertices, test_vertices, train_answers, test_answers, class_labels
#
#
# def __load_psb_vertices(f_name, dir=path_res_3d_psb, is_test=False):
#     if is_test:
#         np_dir = path_res_numpy_psb_test
#     else:
#         np_dir = path_res_numpy_psb_train
#
#     np_path = np_dir + "/" + f_name.split(".")[0] + ".npy"
#
#     # キャッシュがあれば、それを返す
#     if os.path.exists(np_path):
#         return np.load(np_path)
#
#     vertices, faces = parse_off(dir + "/" + f_name)
#
#     return vertices
#
#
# def load_psb_boxels(degree, is_mixed=False, test_size=0.2):
#     train_label_info, test_label_info, class_labels = __psb_label_info(degree)
#
#     train_boxels = []
#     test_boxels = []
#     train_anewers = []
#     test_answers = []
#
#     for label, ids in train_label_info.items():
#         for id in ids:
#             path = path_res_numpy_boxel_psb_train + "/" + str(id) + ".npy"
#             train_boxels.append(np.load(path))
#         train_anewers += [class_labels.index(label)] * len(ids)
#
#     for label, ids in test_label_info.items():
#         for id in ids:
#             path = path_res_numpy_boxel_psb_test + "/" + str(id) + ".npy"
#             test_boxels.append(np.load(path))
#         test_answers += [class_labels.index(label)] * len(ids)
#
#     if is_mixed:
#         x = train_boxels + test_boxels
#         y = train_anewers + test_answers
#         return train_test_split(x, y, test_size=test_size) + [class_labels]
#
#     return train_boxels, test_boxels, train_anewers, test_answers, class_labels
#
#
# def save_psb_vertices(vertices, id, is_test=False):
#     assert vertices.shape[-1] == 3
#     if is_test:
#         np.save(path_res_numpy_psb_test + "/" + str(id), vertices)
#     else:
#         np.save(path_res_numpy_psb_train + "/" + str(id), vertices)
#
#
# def save_psb_boxel(boxel, id, is_test):
#     assert boxel.ndim == 3
#     if is_test:
#         np.save(path_res_numpy_boxel_psb_test + "/" + str(id), boxel)
#     else:
#         np.save(path_res_numpy_boxel_psb_train + "/" + str(id), boxel)
#
#
# def __psb_label_info(cls, degree, train_name='train.cla', test_name='test.cla'):
#     train_info, train_tree = parse_cla(
#         path_res_3d_psb_classifier + "/" + train_name)
#     test_info, test_tree = parse_cla(
#         path_res_3d_psb_classifier + "/" + test_name)
#
#     classes = set()
#
#     def new_info(info_old, tree):
#         info_new = OrderedDict()
#         for label, ids in info_old.items():
#             # label info
#             new_label = tree.parent(label, degree).name
#             info_ids = info_new.get(new_label, [])
#             info_ids.extend(ids)
#             info_new[new_label] = info_ids
#             # class info
#             classes.add(new_label)
#         return info_new
#
#     train_info_new = new_info(train_info, train_tree)
#     test_info_new = new_info(test_info, test_tree)
#
#     return train_info_new, test_info_new, list(classes)
#

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
