# coding:utf-8

import ConfigParser
import collections
import stringutil
import numpy as np
import warnings
import itertools

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
    with open(off_file) as f:
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


def parse_obj(obj_file):
    with open(obj_file) as f:

        vertices = []

        for line in f:

            split = line.split(' ')

            if split[0] == 'v':
                p = list(map(float, split[1:]))
                vertices.append(p)
            elif split[0] == 'vt':
                warnings.warn(
                    "vt row's of .obj file are ignored."
                    " if you want to get vt row's, please edit \'parse.py\'")
                break
            elif split[0] == 'vn':
                warnings.warn(
                    "vn row's of .obj file are ignored."
                    " if you want to get f row's, please edit \'parse.py\'")
                break
            elif split[0] == 'f':
                warnings.warn(
                    "f row's of .obj file are ignored. "
                    "if you want to get f row's, please edit \'parse.py\'")
                break

    return vertices


def parse_cla(cla_file):
    # クラス階層情報を保持するツリー
    tree = ClaTree('0')

    # クラスラベルと属するデータIDのマップ
    classifier = {}

    with open(cla_file) as f:
        if "PSB" not in f.readline():
            raise IOError("file must be \"cla\" format file.")

        n_class, n_data = map(int, f.readline().split(' '))

        while True:
            line = f.readline()

            if line == '':
                break

            split_line = line.split(' ')

            if len(split_line) == 3:
                # ツリーへクラスを登録
                name, parent_name, n = split_line
                tree.add(name, parent_name)

                # 葉ノードの場合、データIDリストを取得
                if int(n) > 0:
                    ids = [int(f.readline()) for i in xrange(int(n))]
                    classifier.setdefault(name, ids)

    return classifier, tree


class ClaTree(object):
    def __init__(self, root_name):
        self.root = self.ClaNode(root_name, None, 0)

    def __str__(self):
        return self.root.__str__()

    def add(self, name, parent_name):
        self.root.add(name, parent_name, 1)

    def parent(self, name, degree):
        node = self.root.search(name)
        if node.degree <= degree:
            return node
        return node.get_parent(degree)

    class ClaNode(object):
        def __init__(self, name, parent, degree, last_node=False):
            self.name = name
            self.parent = parent
            self.children = []
            self.degree = degree
            self.last_node = last_node

        def __str__(self):
            string = self.name
            if self.last_node:
                edge = '\n' + '|   ' * (self.degree - 1) + '    ∟---'
            else:
                edge = '\n' + ('|   ' * self.degree) + '∟---'
            for c in self.children:
                string += edge + c.__str__()
            return string

        def add(self, name, parent_name, degree):
            if parent_name == self.name:
                if len(self.children) > 0:
                    self.children[-1].last_node = False
                node = self.__class__(name, self, degree, True)
                self.children.append(node)
            else:
                for c in self.children:
                    c.add(name, parent_name, degree + 1)

        def search(self, name):
            if self.name == name:
                return self

            for c in self.children:
                node = c.search(name)
                if node is not None:
                    return node
            return None

        def leaf(self):
            if len(self.children) == 0:
                return [self]
            else:
                leaves = []
                for c in self.children:
                    leaves.extend(c.leaf())
                return leaves

        def get_parent(self, degree):
            if self.parent is None:
                return None
            elif self.parent.degree <= degree:
                return self.parent
            else:
                return self.parent.get_parent(degree)

