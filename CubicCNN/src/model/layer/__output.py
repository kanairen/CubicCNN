# coding: utf-8

class OutputLayerInterface(object):
    def cost(self, predict, y):
        raise NotImplementedError

    def error(self, predict, y):
        raise NotImplementedError
