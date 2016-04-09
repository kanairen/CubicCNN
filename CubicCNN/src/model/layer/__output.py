#!/usr/bin/env python
# coding: utf-8

class OutputLayerInterface(object):
    def cost(self, input, answer, is_train):
        raise NotImplementedError

    def error(self, input, answer, is_train):
        raise NotImplementedError
