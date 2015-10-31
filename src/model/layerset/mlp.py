# coding:utf-8

import six
from theano import tensor as T, function, pp

__author__ = 'ren'


class MLP(object):
    def __init__(self, **layers):

        self.inputs_symbol = T.fmatrix('inputs')
        self.answers_symbol = T.lvector('answers')

        self.layers = []
        for name, layer in sorted(six.iteritems(layers)):
            # assert type(layer) == Layer
            setattr(self, name, layer)
            self.layers.append(layer)

        self.output = self.inputs_symbol
        for layer in self.layers:
            self.output = layer.output(self.output)

    def forward(self, inputs, answers, updates=None, givens={}):
        if updates is None:
            updates = self.update()

        f = function(inputs=[self.inputs_symbol, self.answers_symbol],
                     outputs=self.accuracy(self.output, self.answers_symbol),
                     updates=updates,
                     givens=givens)
        # print "arg:",T.mean(T.eq(self.softmax_argmax(self.output),self.answers_symbol)).eval({self.inputs_symbol:inputs,self.answers_symbol:answers})
        print self.output.eval({self.inputs_symbol:inputs})
        # print "arg:",self.softmax_argmax(self.output).eval({self.inputs_symbol:inputs})
        # print "answer:",answers
        return f(inputs, answers)

    def update(self, learning_rate=0.01):
        cost = self.negative_log_likelihood(self.output, self.answers_symbol)
        updates = []
        for layer in self.layers:
            update = layer.update(cost,learning_rate)
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
