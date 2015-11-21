# coding:utf-8

import ConfigParser
import collections
import string
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
            if string.isinteger(value):
                value = int(value)
            elif string.isfloat(value):
                value = float(value)
            elif string.islist(value):
                value = string.tolist(value)
            elif string.istuple(value):
                value = string.totuple(value)

            items.setdefault(option, value)

        item_set.setdefault(section, items)

    return item_set


def parse_off(off_file):
    with file(off_file) as f:
        # 一行目はファイルフォーマット名
        if "OFF" not in f.readline():
            raise IOError("psb file must be \"off\" format file.")

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
