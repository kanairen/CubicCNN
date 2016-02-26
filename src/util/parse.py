# coding:utf-8

import ConfigParser
import collections
import string
import struct
import numpy as np
import warnings

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
            c, r, g, b = map(int, f.readline().rstrip().split(' '))
            faces.append([c, r, g, b])
        faces = np.array(faces)

        return vertices, faces


def parse_obj(obj_file):
    """
    objファイルを読み込み、頂点情報、法線情報、面情報を取得
    :param file_path: ファイルパス
    :return: 頂点リスト、法線リスト、面リスト
    """

    # TODO vt属性の読み込み
    warnings.warn("vt row's of .obj file are ignored.",
                  "if you want to get vt row's, please edit \'parse.py\.'")

    with open(obj_file) as f:
        lines = filter(lambda x: len(x) > 0,
                       [line.strip().split() for line in f.readlines()])
        vertices = [list(map(float, line[1:])) for line in lines if
                    line[0] == 'v']
        normals = [list(map(float, line[1:])) for line in lines if
                   line[0] == 'vn']
        faces = [list(map(int, line[1:])) for line in lines if line[0] == 'f']

    return vertices, normals, faces


def parse_cla(cla_file):
    # クラス階層情報を保持するツリー
    tree = ClaTree('0')

    # クラスラベルと属するデータIDのマップ
    classifier = collections.OrderedDict()

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
                return self
            elif self.degree <= degree:
                return self
            else:
                return self.parent.get_parent(degree)


def parse_binvox(binvox_file, show_params=False):
    """

    .binvoxファイルを読み込み、3Dボクセル配列を返す

    :param binvox_file: PATH含むファイル名
    :return: 3Dボクセル配列
    """

    with open(binvox_file, mode='rb') as f:

        # binvox 1
        binvox = f.readline().strip()

        # 分割数
        dim = tuple(map(int, f.readline().strip().split()[1:]))

        # 標準化の際の平行移動
        trans = tuple(map(float, f.readline().strip().split()[1:]))

        # 標準化の際のスケール
        scale = float(f.readline().strip().split()[1])

        # data（バイナリスタート）
        data = f.readline()

        if show_params:
            print binvox
            print dim
            print trans
            print scale
            print data

        # ボクセル配列
        array = np.zeros(shape=(dim[0] * dim[1] * dim[2]), dtype=np.uint8)

        # 先頭Index
        head = 0

        while True:
            # 2バイトずつ取り出し
            binaly = f.read(1)
            num = f.read(1)

            # ファイル終端であれば終了
            if binaly == '':
                break

            # 0 or 1
            bin_uc = struct.unpack('B', binaly)[0]
            # bin_ucの連続数
            n_uc = struct.unpack('B', num)[0]

            # 元々0埋めの配列なので、bin_uc==1の時だけ代入
            if bin_uc == 1:
                array[head:head + n_uc] = 1

            # 次の値格納のために、headをn_ucずらす
            head += n_uc

    # 3Dにして返戻
    return np.rollaxis(array.reshape(dim), 2)
