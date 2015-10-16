# coding:utf-8

import six
from theano import tensor as T, function
from src.model.mlp.layer import Layer

__author__ = 'ren'


# TODO 出力層のソフトマックス関数実装
# TODO 逆伝播時の更新式を実装
# TODO 精度出力関数を実装

class MLP(object):
    def __init__(self, **layers):

        self.inputs_symbol = T.fmatrix('inputs')

        self.layers = []
        for name, layer in sorted(six.iteritems(layers)):
            assert type(layer) == Layer
            setattr(self, name, layer)
            self.layers.append(layer)

    def chain(self):
        prev_inputs = self.inputs_symbol
        for layer in self.layers:
            layer.inputs = prev_inputs
            prev_inputs = layer.output()

    def forward(self, inputs, updates=(), givens={}):
        output_layer = self.layers[-1]
        return function(inputs=[self.inputs_symbol],
                        outputs=output_layer.output(),
                        updates=updates,
                        givens=givens)(inputs)

    @staticmethod
    def accuracy(pred, answer):
        assert len(pred) == len(answer)
        return function(T.mean(T.eq(pred, answer)))
