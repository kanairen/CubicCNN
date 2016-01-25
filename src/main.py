# coding:utf-8


import sys
import pprint
from src.client.recognizer.image_recognizer import image_recognition
from src.client.recognizer.shape_recognizer import shape_recognition
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

    recognizer = kwarg.pop("recognizer")

    if recognizer == 'image':
        # 画像認識実験
        image_recognition(**kwarg)

    elif recognizer == 'shape':
        # 三次元形状認識実験
        shape_recognition(**kwarg)


if __name__ == '__main__':
    main()
