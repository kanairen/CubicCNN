# coding:utf-8

import six
from theano import tensor as T, function
from src.model.mlp.layer import Layer
from src.util.sequence import joint_dict

__author__ = 'ren'


# TODO 出力層のソフトマックス関数実装 30min(14:40)
# TODO 逆伝播時の更新式を実装 30min(15:15)
# TODO 精度出力関数を実装 30min(15:50)

class MLP(object):
    def __init__(self, **layers):

        self.inputs_symbol = T.fmatrix('inputs')
        self.answers_symbol = T.lvector('answers')

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

    def forward(self, inputs, answers, updates=(), givens={}):
        output_layer = self.layers[-1]
        return function(inputs=[self.inputs_symbol, self.answers_symbol],
                        outputs=self.accuracy(output_layer.output(),
                                              self.answers_symbol),
                        updates=updates,
                        givens=givens)(inputs, answers)

    @staticmethod
    def softmax(x):
        return T.nnet.softmax(x)

    @classmethod
    def softmax_argmax(cls, x):
        return T.argmax(cls.softmax(x), axis=1)

    @classmethod
    def accuracy(cls, x, answers):
        return T.mean(T.eq(cls.softmax_argmax(x), answers))


