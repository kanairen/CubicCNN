# coding:utf-8

from src.model.mlp.layer import Layer

__author__ = 'ren'


class MLP(object):
    def __init__(self, n_in, n_hidden, n_out):
        self.l1 = Layer(n_in=n_in, n_out=n_hidden)
        self.l2 = Layer(n_in=n_hidden, n_out=n_out)

    def forward(self):
        pass


