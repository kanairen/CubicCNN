# coding:utf-8

import ConfigParser
import collections
import stringutil
import numpy as np

__author__ = 'ren'


def parse_ini(ini_file):
    ini = ConfigParser.SafeConfigParser()
    ini.read(ini_file)

    item_set = collections.OrderedDict()
    for section in ini.sections():
        items = {}

        for option in ini.options(section):
            value = ini.get(section, option)
            if stringutil.isinteger(value):
                value = int(value)
            elif stringutil.isfloat(value):
                value = float(value)
            elif stringutil.islist(value):
                value = stringutil.tolist(value)
            elif stringutil.istuple(value):
                value = stringutil.totuple(value)

            items.setdefault(option, value)

        item_set.setdefault(section, items)

    return item_set


def parse_off(off_file):
    with file(off_file) as f:
        # 一行目はファイルフォーマット名
        if "OFF" not in f.readline():
            raise IOError("file must be \"off\" format file.")

        n_vertices, n_faces, n_edges = map(int, f.readline().split(' '))

        vertices = []
        for i in xrange(n_vertices):
            x, y, z = map(float, f.readline().split(' '))
            vertices.append([x, y, z])
        vertices = np.array(vertices)

        faces = []
        for i in xrange(n_faces):
            c, r, g, b = map(int, f.readline().split(' '))
            faces.append([c, r, g, b])
        faces = np.array(faces)

        return vertices, faces


def parse_cla(cla_file):
    class_tree = {}
    class_history = []

    with file(cla_file) as f:
        if "PSB" not in f.readline():
            raise IOError("file must be \"cla\" format file.")

        n_class, n_data = map(int, f.readline().split(' '))

        current_tree = class_tree

        while True:
            split = f.readline().split(' ')
            if split == '\n':
                continue

            cls, parent, n = split
            n = int(n)

            if n > 0:
                ids = []
                for i in xrange(n):
                    ids.append(int(f.readline()))
                current_tree.setdefault(cls, ids)
            else:
                class_history.append(cls)
                current_tree.setdefault(cls, {})
                current_tree = current_tree.get(cls)

            parent_node = parent
