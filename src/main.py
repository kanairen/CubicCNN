# coding:utf-8

import sys
from src.client.image_recognition import image_recognition
from src.client.solid_recognition import solid_recognition

__author__ = 'ren'

if __name__ == '__main__':

    # コマンドライン引数
    param = sys.argv
    # クライアントタイプ
    client = param[1]

    if client == 'image_recognition':
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

        print "data_type : ", data_type
        print "n_iter : ", n_iter
        print "n_batch : ", n_batch
        print "show_batch_accuracies : ", show_batch_accuracies
        print "save_batch_accuracies : ", save_batch_accuracies

        # 画像認識実験
        image_recognition(data_type, n_iter, n_batch, show_batch_accuracies,
                          save_batch_accuracies)
    elif client == 'solid_recognition':
        data_type = param[2]
        solid_recognition(data_type=data_type)

