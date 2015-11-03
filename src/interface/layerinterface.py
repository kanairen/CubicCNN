# coding:utf-8

__author__ = 'ren'


class LayerInterface(object):
    def __init__(self):
        pass

    def update(self):
        raise NotImplementedError(
            "layer objects should be implemented 'update'.")

    def output(self):
        raise NotImplementedError(
            "layer objects should be implemented 'output'.")
