# coding:utf-8

import six
from theano import tensor as T

__author__ = 'ren'


class MLP(object):
    def __init__(self, **layers):

        self.names = []
        print "name"
        for name, layer in six.iteritems(layers):
            print name
            setattr(self, name, layer)
            self.names.append(name)

    def chain(self):
        prev = T.fmatrix('inputs')
        for name in self.names:
            layer = getattr(self, name)
            layer.inputs = prev.output
            prev = layer

    def forward(self, inputs, updates=(), givens={}):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs, updates=updates, givens=givens)
        return outputs
