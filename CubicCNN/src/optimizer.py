#!/usr/bin/env python
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

    def optimize(self, n_iter, n_batch, on_optimized=None,
                 is_print_enabled=True):
        x_train, x_test, y_train, y_test = self.data.data()

        n_batch_train = len(x_train) / n_batch
        n_batch_test = len(x_test) / n_batch

        sum_error_all = 0

        for iter in xrange(n_iter):
            for j in xrange(n_batch):

                b_x_train = x_train[j * n_batch_train:(j + 1) * n_batch_train]
                b_x_test = x_test[j * n_batch_test:(j + 1) * n_batch_test]
                b_y_train = y_train[j * n_batch_train:(j + 1) * n_batch_train]
                b_y_test = y_test[j * n_batch_test:(j + 1) * n_batch_test]

                cost = self._train(b_x_train, b_y_train)
                error = self._test(b_x_test, b_y_test)
                error_all = self._test(x_test, y_test)
                sum_error_all += error_all

                if on_optimized is not None:
                    on_optimized(cost, error)
                if is_print_enabled:
                    print "{}th iteration / {}th batch".format(iter, j)
                    print "cost:{}".format(cost)
                    print "error:{}".format(error)
                    print "error all:{}".format(error_all)
                    print "error all average:{}".format(
                        sum_error_all / (n_batch * iter + j + 1))

    def _update(self, learning_rate):
        grads = T.grad(self.model.cost(True), self.model.params)
        updates = [(p, p - learning_rate * g) for p, g in
                   zip(self.model.params, grads)]
        return updates
