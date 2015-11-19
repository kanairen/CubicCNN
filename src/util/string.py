# coding:utf-8

import re

__author__ = 'ren'


def isbool(string):
    return string == 'True' or string == 'False'


def isinteger(string):
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


def _isset(string, first, last):
    return string.find(first) == 0 and string.find(last) == len(string) - 1


def tolist(string):
    return _toset(string, '[', ']')


def totuple(string):
    return _toset(string, '(', ')')


def _toset(string, first, last):
    str_list = string.replace(first, "").replace(last, '').replace('.', '')
    return list(map(int, str_list))

