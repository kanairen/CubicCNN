#!/usr/bin/env python
# coding: utf-8

import time
from theano import function, tensor as T
from data.__cls import Data
from model.model import Model
from result import Result


class Optimizer(object):
    KEY_TRAIN_COST = 'cost'
    KEY_TRAIN_TIME = 'train_time'
    KEY_TEST_TOTAL_ERROR = 'total_error'
    KEY_TEST_TOTAL_ERROR_AVERAGE = 'total_error_average'
    KEY_TEST_TOTAL_ERROR_TIME = 'total_test_time'
    KEY_TEST_BATCH_ERROR = 'batch_error'
    KEY_TEST_BATCH_ERROR_AVERAGE = 'batch_error_average'
    KEY_TEST_BATCH_ERROR_TIME = 'batch_test_time'

    def __init__(self, data, model, learning_rate=0.01):
        assert isinstance(data, Data)
        assert isinstance(model, Model)
        self.data = data
        self.model = model
        self.result = Result()

        self._train = function(
            inputs=(self.model.input_symbol, self.model.answer_symbol),
            outputs=self.model.cost(True),
            updates=self._update(learning_rate))

        self._test = function(
            inputs=(self.model.input_symbol, self.model.answer_symbol),
            outputs=self.model.error(False),
            updates=[])

    def optimize(self, n_iter, n_batch, is_total_test_enabled=True,
                 is_print_enabled=True):
        x_train, x_test, y_train, y_test = self.data.data()

        bs_train = len(x_train) / n_batch
        bs_test = len(x_test) / n_batch

        sum_error_all = 0.

        for i in xrange(n_iter):
            batch_sum_error = 0.
            for j in xrange(n_batch):

                b_x_train = x_train[j * bs_train:(j + 1) * bs_train]
                b_x_test = x_test[j * bs_test:(j + 1) * bs_test]
                b_y_train = y_train[j * bs_train:(j + 1) * bs_train]
                b_y_test = y_test[j * bs_test:(j + 1) * bs_test]

                # train
                start = time.clock()
                cost = self._train(b_x_train, b_y_train)
                train_time = time.clock() - start

                # batch test
                start = time.clock()
                batch_error = self._test(b_x_test, b_y_test)
                batch_test_time = time.clock() - start

                batch_sum_error += batch_error

                # 結果の保存
                self.result.add_all(
                    ((self.KEY_TRAIN_TIME, train_time),
                     (self.KEY_TEST_BATCH_ERROR_TIME, batch_test_time),
                     (self.KEY_TRAIN_COST, cost),
                     (self.KEY_TEST_BATCH_ERROR, batch_error),
                     (self.KEY_TEST_BATCH_ERROR_AVERAGE,
                      batch_sum_error / (j + 1))))

                # total test
                if is_total_test_enabled:
                    start = time.clock()
                    total_error = self._test(x_test, y_test)
                    total_error_time = time.clock() - start

                    sum_error_all += total_error

                    self.result.add_all(
                        ((self.KEY_TEST_TOTAL_ERROR, total_error),
                         (self.KEY_TEST_TOTAL_ERROR_TIME, total_error_time),
                         (self.KEY_TEST_TOTAL_ERROR_AVERAGE,
                          sum_error_all / (n_batch * i + j + 1))))

                # output
                if is_print_enabled:
                    print '\n{}th iteration / {}th batch'.format(i + 1, j + 1)
                    for l, a in self.result.results.items():
                        m_format = "{}: {}s" if 'time' in l else "{}: {}"
                        print m_format.format(l.replace('_', ' '), a[-1])

    def _update(self, learning_rate):
        grads = T.grad(self.model.cost(True), self.model.params)
        updates = [(p, p - learning_rate * g) for p, g in
                   zip(self.model.params, grads)]
        return updates
