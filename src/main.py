# coding:utf-8

import sys
from src.client.recognizer.image_recognizer import image_recognition
from src.client.recognizer.solid_recognizer import psb_recognition
from src.helper.decorator import client

__author__ = 'ren'


@client
def main():
    # コマンドライン引数
    param = sys.argv
    # クライアントタイプ
    client = param[1]
    # データタイプ
    data_type = param[2]
    # 学習繰り返し数
    n_iter = int(param[3])
    # バッチ数
    n_batch = int(param[4])
    # バッチごとの精度を表示するかどうか
    show_batch_accuracies = bool(param[5])
    # バッチごとの精度を保存するかどうか
    save_batch_accuracies = bool(param[6])

    print "recognizer : ", client
    print "data_type : ", data_type
    print "n_iter : ", n_iter
    print "n_batch : ", n_batch
    print "show_batch_accuracies : ", show_batch_accuracies
    print "save_batch_accuracies : ", save_batch_accuracies

    if client == 'image_recognition':
        # 画像認識実験
        image_recognition(data_type, n_iter, n_batch,
                          show_batch_accuracies, save_batch_accuracies)

    elif client == 'solid_recognition':
        # 三次元形状認識実験
        if data_type == 'psb':
            psb_recognition(n_iter, n_batch, show_batch_accuracies,
                            save_batch_accuracies)


if __name__ == '__main__':
    main()
