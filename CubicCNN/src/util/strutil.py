#!/usr/bin/env python
# coding:utf-8

import re

__author__ = 'ren'


def isbool(string):
    return string == 'True' or string == 'False'


def isinteger(string):
    if len(string) > 0 and string[0] == '-':
        string = string.replace('-', '')
    if string.isdigit():
        if len(string) > 1 and string[0] == '0':
            return False
        else:
            return True
    return False


def isfloat(string):
    return bool(re.compile("\d*\.\d*\Z").match(string))


def islist(string):
    return _isset(string, '[', ']')


def istuple(string):
    return _isset(string, '(', ')')


def isnone(string):
    return string == 'None'


def _isset(string, start, end):
    return string.startswith(start) and string.endswith(end)


def tolist(string):
    return _toset(string, '[', ']')


def totuple(string):
    return _toset(string, '(', ')')


def _toset(string, start, end):
    str_list = string.replace(start, '').replace(end, '').split(',')
    seq = []
    for str in str_list:
        if isinteger(str):
            seq.append(int(str))
        elif isfloat(str):
            seq.append(float(str))
        else:
            seq.append(str)

    if istuple(string):
        seq = tuple(seq)

    return seq


def toany(string):
    if isbool(string):
        return bool(string)
    elif isinteger(string):
        return int(string)
    elif isfloat(string):
        return float(string)
    elif islist(string):
        return tolist(string)
    elif istuple(string):
        return totuple(string)
    elif isnone(string):
        return None
    else:
        return string
