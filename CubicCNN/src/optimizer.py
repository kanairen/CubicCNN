# coding: utf-8

from theano import function, tensor as T
from data.__cls import Data
from model.model import Model


class Optimizer(object):
    def __init__(self, data, model, learning_rate=0.01):
        assert isinstance(data, Data)
        assert isinstance(model, Model)
        self.data = data
        self.model = model

        self._train = function(
            inputs=(self.model.input_symbol, self.model.answer_symbol),
            outputs=self.model.cost(True),
            updates=self._update(learning_rate))

        self._test = function(
            inputs=(self.model.input_symbol, self.model.answer_symbol),
            outputs=self.model.error(False),
            updates=[])

    def optimize(self, n_iter, on_optimized=None, is_print_enabled=True):
        x_train, x_test, y_train, y_test = self.data.data()

        for i in xrange(n_iter):
            cost = self._train(x_train, y_train)
            error = self._test(x_test, y_test)
            if on_optimized is not None:
                on_optimized(cost, error)
            if is_print_enabled:
                print "cost:{}".format(cost)
                print "error:{}".format(error)

    def _update(self, learning_rate):
        grads = T.grad(self.model.cost(True), self.model.params)
        updates = [(p, p - learning_rate * g) for p, g in
                   zip(self.model.params, grads)]
        return updates
