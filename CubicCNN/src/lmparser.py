#!/usr/bin/env python
# coding: utf-8

from collections import OrderedDict
from model.layer.__hidden import HiddenLayer
from model.layer.__conv import ConvLayer2d
from model.layer.__pool import MaxPoolLayer2d
from model.layer.__softmax import SoftMaxLayer
from util import strutil

# lmparser.py called as singleton object.

# key
KEY_DATA = 'data'

KEY_HIDDEN = 'hidden'
KEY_CONV = 'conv'
KEY_POOL = 'pool'
KEY_SOFTMAX = 'softmax'


def parse(lm_file):
    items = []
    with open(lm_file) as f:
        while True:
            line = f.readline()
            if line == '':
                break
            elif line[0] == '#':
                continue

            sline = line.strip()
            if sline == '':
                continue
            elif '(' == sline[-1]:
                item_type = sline.strip('(')
                attr_lines = []
                line = f.readline().strip()
                while ')' != line:
                    attr_lines.append(line)
                    line = f.readline().strip()
                items.append((item_type, attr_lines))

    layers = __parse_layers(items)
    print layers


def __parse_layers(items):
    layers = []
    for index, item in enumerate(items):
        item_type, attr_lines = item
        kwargs = parse_attrs(attr_lines)
        if item_type == KEY_HIDDEN:
            layer = HiddenLayer(layer_id=index, **kwargs)
        elif item_type == KEY_CONV:
            layer = ConvLayer2d(layer_id=index, **kwargs)
        elif item_type == KEY_POOL:
            layer = MaxPoolLayer2d(layer_id=index, **kwargs)
        elif item_type == KEY_SOFTMAX:
            layer = SoftMaxLayer(layer_id=index, **kwargs)
        else:
            continue
        layers.append(layer)
    return layers


def parse_attrs(attr_lines):
    kwargs = {}
    for line in attr_lines:
        k, v = line.strip().split('=')
        kwargs[k] = strutil.toany(v)
    return kwargs


if __name__ == '__main__':
    import os

    print os.path.abspath(os.curdir)
    print parse('../../temp.lm')
