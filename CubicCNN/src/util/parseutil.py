#!/usr/bin/env python
# coding: utf-8

import struct
import collections
import numpy as np


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
    return array.reshape(dim)


def parse_off(off_file):
    """
    .offファイルを読み込み、3Dボクセル配列を返す
    :type off_file: str
    :param off_file: PATH含むファイル名
    :rtype tuple(numpy.ndarray)
    :returns 頂点座標のリストと面リスト
    """

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


# TODO ClaTreeを使わない設計にする
def parse_cla(cla_file):
    """
    .claファイルを読み込み、所属クラスデータを返す
    :type cla_file: str
    :param cla_file: PATH含むファイル名
    :rtype collections.OrderedDict
    :return: クラスラベル:属するデータの辞書
    """
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

    return classifier


class ClaTree(object):
    """
    .claファイル中のクラス階層を表現するクラス
    """

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


def parse_vxl(vxl_file):
    """
    .vxlファイルを読み込み、ボクセルデータを返す
    :type vxl_file: str
    :param vxl_file: PATH含むファイル名
    :rtype numpy.ndarray
    :return ボクセルデータ
    """

    with open(vxl_file, 'rb') as f:
        # comment
        comment = f.readline()

        # dim: ボクセルxyz軸の分解能
        dim = map(int, f.readline().strip().split(' ')[1:])

        # data: バイナリ開始行
        data = f.readline()

        # ボクセル配列
        array = np.zeros(shape=(dim[0] * dim[1] * dim[2]), dtype=np.uint8)

        # 先頭インデックス
        head = 0

        while True:
            # 2バイトずつ取り出し
            binaly = f.read(1)
            num = f.read(1)

            # ファイル終端であれば終了
            if binaly == '':
                break

            # ボクセル値
            bin_uc = struct.unpack('B', binaly)[0]
            # bin_ucの連続数
            n_uc = struct.unpack('B', num)[0]

            # ボクセル内の値が非ゼロの場合、n_ucボクセル分だけbin_ucの値で埋めていく
            if bin_uc > 0:
                array[head:head + n_uc] = bin_uc

            # 次の値格納のために、headをn_ucずらす
            head += n_uc

    return array.reshape(dim)

