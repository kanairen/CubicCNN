# coding:utf-8

__author__ = 'ren'


class LayerInterface(object):
    def __init__(self):
        pass

    def update(self, cost, learning_rate):
        raise NotImplementedError(
            "layer objects should be implemented 'update'.")

    def output(self, inputs_symbol):
        raise NotImplementedError(
            "layer objects should be implemented 'output'.")
