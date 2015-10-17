# coding:utf-8

import six
import itertools
from theano import tensor as T, function, shared, config
from src.model.mlp.layer import Layer
from src.util.sequence import joint_dict

__author__ = 'ren'


# 出力層のソフトマックス関数実装 30min(14:40) fin
# 精度出力関数を実装 30min(15:05) fin
# TODO 逆伝播時の更新式を実装 30min(15:30)
# TODO コスト関数
# TODO 微分
# TODO 更新式
# TODO 関数呼び出し

class MLP(object):
    def __init__(self, **layers):

        self.inputs_symbol = T.fmatrix('inputs')
        self.answers_symbol = T.lvector('answers')

        self.layers = []
        for name, layer in sorted(six.iteritems(layers)):
            assert type(layer) == Layer
            setattr(self, name, layer)
            self.layers.append(layer)

        self.params = list(itertools.chain(*[l.params for l in self.layers]))

    def chain(self):
        prev_inputs = self.inputs_symbol
        for layer in self.layers:
            layer.inputs = prev_inputs
            prev_inputs = layer.output()

    def forward(self, inputs, answers, updates=None, givens={}):
        if updates is None:
            updates = self.update()

        output_layer = self.layers[-1]
        return function(inputs=[self.inputs_symbol, self.answers_symbol],
                        outputs=self.accuracy(output_layer.output(),
                                              self.answers_symbol),
                        updates=updates,
                        givens=givens)(inputs, answers)

    def update(self, learning_rate=0.1):
        output_layer = self.layers[-1]
        cost = self.negative_log_likelihood(output_layer.output(),
                                            self.answers_symbol)
        grads = T.grad(cost, self.params)
        return [(p, p - learning_rate * g) for p, g in zip(self.params, grads)]

    @staticmethod
    def softmax(x):
        return T.nnet.softmax(x)

    @classmethod
    def softmax_argmax(cls, x):
        return T.argmax(cls.softmax(x), axis=1)

    @classmethod
    def accuracy(cls, x, answers):
        return T.mean(T.eq(cls.softmax_argmax(x), answers))

    @classmethod
    def negative_log_likelihood(cls, x, y):
        return -T.mean(T.log(cls.softmax(x))[T.arange(y.shape[0]), y])
