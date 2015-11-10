# coding:utf-8

import six
from theano import tensor as T, function

__author__ = 'ren'


class MLP(object):
    def __init__(self, **layers):

        self.inputs_symbol = T.fmatrix('inputs')
        self.answers_symbol = T.lvector('answers')

        self.layers = []
        self.output = self.inputs_symbol
        self.regularization_term = 0
        for name, layer in sorted(six.iteritems(layers)):
            setattr(self, name, layer)
            self.layers.append(layer)
            self.output = layer.output(self.output)
            self.regularization_term += abs(layer.W).sum()

    def forward(self, inputs, answers, updates=None, givens={}):

        if updates is None:
            updates = self.update()

        f = function(inputs=[self.inputs_symbol, self.answers_symbol],
                     outputs=self.accuracy(self.output, self.answers_symbol),
                     updates=updates,
                     givens=givens)

        return f(inputs, answers)

    def update(self, learning_rate=0.01, regularization_rate=0.001):
        cost = self.negative_log_likelihood(self.output,
                                            self.answers_symbol) + regularization_rate * self.regularization_term
        updates = []
        for layer in self.layers:
            update = layer.update(cost, learning_rate)
            if update is not None:
                updates.extend(update)
        return updates

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
