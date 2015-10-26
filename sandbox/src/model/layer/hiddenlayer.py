# coding=utf-8

from sandbox.src.util.activation import *

__author__ = 'kanairen'

rnd = np.random.RandomState(1111)


class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, W=None, b=None, activation=relu,
                 dtype=np.float32):
        # 入力値
        self.input = input

        # 重み行列
        if W is None:
            W = np.asarray(rnd.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                       high=np.sqrt(6. / (n_in + n_out)),
                                       size=(n_in, n_out)),
                           dtype=dtype)
        self.W = W

        # バイアスベクトル
        if b is None:
            b = np.zeros((n_out,), dtype=dtype)
        self.b = b

        # 活性化関数
        if activation is None:
            activation = lambda x: x
        self.activation = activation

        # 活性化関数の導関数
        if activation is sigmoid:
            d_activation = d_sigmoid
        elif activation is relu:
            d_activation = d_relu
        else:
            raise ValueError(
                'derivative function of specified activation is not supported.')

        self.d_activation = d_activation

        # 誤差逆伝播に用いるデルタ
        self.delta = None

    def output(self):
        return self.forward(self.input)

    def forward(self, input):

        linear_output = np.dot(input, self.W) + self.b

        return self.activation(linear_output)

    def backward(self, prev_layer, learning_rate=0.1):

        delta = self.d_activation(prev_layer.input) * np.dot(prev_layer.delta,
                                                             prev_layer.W.T)
        dW = learning_rate * np.dot(self.input.T, delta)
        db = learning_rate * np.mean(delta, axis=0)

        self.W += dW
        self.b += db
        self.delta = delta

