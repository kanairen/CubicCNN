# coding:utf-8

import numpy as np

from src.helper.decorator import client
from src.data.psb import PSB
from src.data.image import Image
from src.helper.visualize import plot_2d
from src.model.layer.conv import ConvLayer2d
from src.model.layer.pool import PoolLayer
from src.model.layerset.mlp import MLP
from src.util.config import path_res_2d_pattern, path_res_numpy_array
from src.util.time import ymdt


# TODO プレーン
# TODO スポーク

# TODO テストデータシャッフル 15min(18:35)
# TODO 誤りデータの表示と誤り例（深層学習　p）1h
# TODO Dropoutの実装 1h
# TODO モデルのサーフェスだけでなく、ソリッドも試す
# TODO EASY-CLASSIFIERの実装（クラス分類数を大まかなものに変更）30min

@client
def cubic_cnn(n_div=50, img_size=(64, 64), is_boxel=False):
    """
    DATA
    """
    print "loading data..."
    if is_boxel:
        # 点群データ
        train_ins, train_ans = PSB.load_boxels(is_test=False)
        test_ins, test_ans = PSB.load_boxels(is_test=True)
    else:
        inputs, answers, images, r_images = Image.image_set(path_res_2d_pattern,
                                                            Image.TRANS,
                                                            size=img_size)
        train_ins, test_ins, train_ans, test_ans, perm = Image.hold_out(
            inputs, answers, train_rate=0.8)

    # numpy配列にしておくと、スライシングでコピーが発生しない
    train_ins = np.array([inputs.flatten() for inputs in train_ins])
    test_ins = np.array([inputs.flatten() for inputs in test_ins])
    train_ans = np.array(train_ans)
    test_ans = np.array(test_ans)

    print "train data : ", len(train_ins)
    print "test data : ", len(test_ins)
    print "classes : ", len(set(train_ans.tolist() + test_ans.tolist()))

    """
    MODEL
    """

    print "preparing models..."

    n_in = n_div ** 3 if is_boxel else img_size

    model = MLP(l1=ConvLayer2d(n_in, in_channel=1, out_channel=1, k_size=3),
                l2=PoolLayer(n_in, in_channel=1, k_size=3))

    """
    TRAIN
    """

    # トレーニング繰り返し回数
    n_iter = 1000

    # バッチ数
    n_batch = 1

    # バッチサイズ
    batch_size_train = len(train_ins) / n_batch
    batch_size_test = len(test_ins) / n_batch

    # 訓練・テストにおける精度の時系列リスト
    train_accuracies = []
    test_accuracies = []

    for i in range(n_iter):

        print "{}st learning...".format(i)

        train_accuracy = 0
        test_accuracy = 0

        # バッチごとに学習
        for j in range(n_batch):
            print "{}st batch...".format(j)
            from_train = j * batch_size_train
            from_test = j * batch_size_test
            to_train = (j + 1) * batch_size_train
            to_test = (j + 1) * batch_size_test

            train_accuracy += model.forward(
                inputs=train_ins[from_train:to_train],
                answers=train_ans[from_train:to_train])
            test_accuracy += model.forward(
                inputs=test_ins[from_test:to_test],
                answers=test_ans[from_test:to_test],
                updates=())

        train_accuracy /= n_batch
        test_accuracy /= n_batch

        print "train : ", train_accuracy
        print "test : ", test_accuracy

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    # グラフの描画
    plot_2d(xlabel="iteration", ylabel="accuracy", ylim=(0, 1),
            train=train_accuracies,
            test=test_accuracies)

    # 精度の保存
    np.save(path_res_numpy_array + "/" + ymdt() + "_traain", train_accuracies)
    np.save(path_res_numpy_array + "/" + ymdt() + "_test", test_accuracies)


if __name__ == '__main__':
    cubic_cnn()
