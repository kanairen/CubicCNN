#!/usr/bin/env python
# coding: utf-8

from itertools import chain
from theano import tensor as T
from layer.__conv import GridLayer2d, GridLayer3d
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

        self.layers = layers

        # 二次元レイヤが先頭の場合、シンボルの形状を変更する
        if isinstance(layers[0], GridLayer2d) and input_symbol.ndim != 4:
            input_symbol = input_symbol.reshape(
                (input_symbol.shape[0], 1, layers[0].input_size[0],
                 layers[0].input_size[1]))
        # 三次元レイヤが先頭の場合、シンボルの形状を変更する
        elif isinstance(layers[0], GridLayer3d) and input_symbol.ndim != 5:
            input_symbol = T.TensorType('float32', (False,) * 5)(name='input')

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
        else:
            assert isinstance(layer, OutputLayerInterface)
        return prev_output

    @staticmethod
    def create_from_ml_file(ml_file):
        raise NotImplementedError

    @staticmethod
    def create_from_layers(layers_gen_func):
        return layers_gen_func()

    def argmax(self, is_train):
        output = self.output_train if is_train else self.output_test
        return T.argmax(output, axis=1)

    def cost(self, is_train):
        output = self.output_train if is_train else self.output_test
        return -T.mean(T.log(output)[T.arange(
            self.answer_symbol.shape[0]), self.answer_symbol])

    def error(self, is_train):
        predict = self.argmax(is_train)
        if predict.ndim != self.answer_symbol.ndim or not self.answer_symbol.dtype.startswith(
                'int'):
            raise TypeError
        return T.mean(T.neq(predict, self.answer_symbol))

    def __str__(self):
        return ''
