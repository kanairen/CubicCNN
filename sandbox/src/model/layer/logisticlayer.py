# coding=utf-8

import numpy as np
from sandbox.src.util.activation import relu, softmax

__author__ = 'kanairen'


class LogisticRegressionLayer(object):
    def __init__(self, input, answer, n_in, n_out):
        self.input = input
        self.answer = answer
        self.W = np.zeros((n_in, n_out))
        self.b = np.zeros((n_out,))
        self.delta = None
        self.activation = softmax

    def forward(self):
        return self.linear_sum(self.input)

    def test(self, input):
        return self.linear_sum(input)

    def linear_sum(self, input):
        return self.activation(np.dot(input, self.W) + self.b)


    def backward(self, learning_rate=0.1, norm1=0.0):
        linear_output = np.dot(self.input, self.W) + self.b
        print linear_output.shape

        delta = self.answer - softmax(linear_output)
        dW = learning_rate * np.dot(self.input.T,
                                    delta) - learning_rate * norm1 * self.W
        db = learning_rate * np.mean(delta, axis=0)

        self.W += dW
        self.b += db
        self.delta = delta

    def negative_log_likelihood(self):
        likelihood = softmax(np.dot(self.input, self.W) + self.b)

        return np.mean(np.sum(
            self.answer * np.log(likelihood) + (1 - self.answer) * np.log(
                1 - likelihood), axis=1))

