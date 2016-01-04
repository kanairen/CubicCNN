# coding:utf-8

import sys
from src.client.recognizer.image_recognizer import image_recognition
from src.client.recognizer.shape_recognizer import shape_recognition
from src.helper.decorator import client

__author__ = 'ren'


@client
def main():
    # コマンドライン引数
    argv = sys.argv

    # クライアントタイプ
    recognizer = argv[1]
    # データタイプ
    data_type = argv[2]
    # 学習繰り返し数
    n_iter = int(argv[3])
    # バッチ数
    n_batch = int(argv[4])
    # データ加工タイプ
    aug_type = argv[5]
    # バッチごとの精度を表示するかどうか
    show_batch_accuracies = bool(argv[6])
    # バッチごとの精度を保存するかどうか
    save_batch_accuracies = bool(argv[7])

    print "recognizer : ", recognizer
    print "data_type : ", data_type
    print "n_iter : ", n_iter
    print "n_batch : ", n_batch
    print "augmentation type : ", aug_type
    print "show_batch_accuracies : ", show_batch_accuracies
    print "save_batch_accuracies : ", save_batch_accuracies

    if recognizer == 'image':
        # 画像認識実験
        image_recognition(data_type, n_iter, n_batch,
                          show_batch_accuracies, save_batch_accuracies)

    elif recognizer == 'solid':
        # 三次元形状認識実験
        shape_recognition(data_type, n_iter, n_batch, aug_type,
                          show_batch_accuracies, save_batch_accuracies)


if __name__ == '__main__':
    main()
