# coding:utf-8

__author__ = 'ren'


def joint_dict(dict1, dict2):
    assert type(dict1) == type(dict2) == dict
    keys = set(dict1.keys() + dict2.keys())

    new_dict = {}
    for key in keys:
        value = dict1.get(key, []) + dict2.get(key, [])
        new_dict.setdefault(key, value)

    return new_dict
