# coding:utf-8


import sys
import pprint
from src.client.classifier.image_classifier import image_classification
from src.client.classifier.shape_classifier import shape_classification
from src.helper.decorator import client
from src.util import string

__author__ = 'ren'


@client
def main():
    # コマンドライン引数
    kwarg = dict((tuple(arg.split("=")) for arg in sys.argv[1:]))

    # 引数のキャスト
    for k, v in kwarg.items():
        if string.isinteger(v):
            kwarg[k] = int(v)
        elif string.isbool(v):
            kwarg[k] = bool(v)
        elif string.isfloat(v):
            kwarg[k] = float(v)
        elif string.islist(v):
            kwarg[k] = string.tolist(v)
        elif string.istuple(v):
            kwarg[k] = string.totuple(v)
        elif string.isnone(v):
            kwarg[k] = None

    pprint.pprint(kwarg)

    classifier = kwarg.pop("classifier")

    if classifier == 'image':
        # 画像認識実験
        image_classification(**kwarg)

    elif classifier == 'shape':
        # 三次元形状認識実験
        shape_classification(**kwarg)


if __name__ == '__main__':
    main()
