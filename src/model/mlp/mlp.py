# coding:utf-8

from src.model.mlp.layer import Layer

__author__ = 'ren'


class MLP(object):
    def __init__(self, n_units, activation=None):

        self.layers = []
        prev_inputs = None
        for i in range(len(n_units) - 1):
            layer = Layer(n_units[i], n_units[i + 1], inputs=prev_inputs,
                          activation=activation)
            self.layers.append(layer)
            prev_inputs = layer.outputs

    def forward(self, inputs, updates=(), givens={}):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs, updates=updates, givens=givens)
        return outputs
