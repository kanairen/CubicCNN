# coding: utf-8

from itertools import chain
from theano import tensor as T
from layer.__base import BaseLayer
from layer.__output import OutputLayerInterface

"""
'Model' means Learning Model in Machine Learning.
"""


class Model(object):
    def __init__(self, layers_gen_func=None, ml_file=None,
                 input_symbol=T.fmatrix('input'),
                 answer_symbol=T.ivector('answer')):
        if ml_file:
            layers = self.create_from_ml_file(ml_file)
        elif layers_gen_func:
            layers = self.create_from_layers(layers_gen_func)
        else:
            raise ValueError

        output_layer = layers[-1]
        assert isinstance(output_layer, OutputLayerInterface)

        self.layers = layers
        self.input_symbol = input_symbol
        self.answer_symbol = answer_symbol
        self.params = list(chain(*[layer.params for layer in layers]))

        # is_trainフラグ切り替えによるドロップアウト適用のため、outputは２通り作る
        # (既に初期化済みのシンボルから各outputを作るため、trainingの結果は双方に反映される)
        self.output_train = self._create_output(True)
        self.output_test = self._create_output(False)

    def _create_output(self, is_train):
        assert isinstance(is_train, bool)
        prev_output = self.input_symbol
        for layer in self.layers:
            prev_output = layer.output(prev_output, is_train)
        return prev_output

    def create_from_ml_file(self, ml_file):
        raise NotImplementedError

    def create_from_layers(self, layers_gen_func):
        layers = layers_gen_func()
        for layer in layers:
            assert isinstance(layer, BaseLayer)
        return layers

    def error(self, is_train):
        predict = self.output_train if is_train else self.output_test
        return self.layers[-1].error(predict, self.answer_symbol)

    def cost(self):
        # cost関数は訓練時のみ使う
        predict = self.output_train
        return self.layers[-1].cost(predict, self.answer_symbol)

    def __str__(self):
        return ''
