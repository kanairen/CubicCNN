# coding:utf-8

__author__ = 'ren'


class BaseLayer(object):
    def __init__(self):
        pass

    def update(self, cost, learning_rate):
        raise NotImplementedError(
            "layer objects should be implemented 'update'.")

    def output(self, inputs_symbol, is_train):
        raise NotImplementedError(
            "layer objects should be implemented 'output'.")
