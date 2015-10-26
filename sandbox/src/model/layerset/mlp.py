# coding=utf-8

from sandbox.src.model.layer.hiddenlayer import HiddenLayer
from sandbox.src.model.layer.logisticlayer import LogisticRegressionLayer

__author__ = 'kanairen'


class MLP(object):
    def __init__(self, input, answer, n_units):

        assert len(n_units) > 1

        # 入力データ
        self.input = input
        # 正解データ
        self.answer = answer

        # 隠れ層の生成
        self.hidden_layers = []

        # レイヤ配列を作る
        prev_input = input
        for i in range(len(n_units) - 2):
            n_in = n_units[i]
            n_out = n_units[i + 1]

            layer = HiddenLayer(prev_input, n_in, n_out)

            self.hidden_layers.append(layer)

            prev_input = layer.output()

        self.last_layer = LogisticRegressionLayer(prev_input, answer,
                                                  n_units[-2],
                                                  n_units[-1])

    def train(self, epoch=100):
        for i in range(epoch):

            # 出力層の勾配を導出
            self.last_layer.backward()

            # 逆伝播
            prev_layer = self.last_layer
            for layer in reversed(self.hidden_layers):
                layer.backward(prev_layer)
                prev_layer = layer


    def test(self, input):
        prev_input = input
        for layer in self.hidden_layers:
            prev_input = layer.forward(prev_input)
        return self.last_layer.test(prev_input)





















