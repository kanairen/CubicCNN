# coding:utf-8

import numpy as np

from src.helper.decorator import client
from src.data.psb import PSB
from src.helper.visualize import plot_2d
from src.helper.image import Image
from src.model.layer.layer import Layer
from src.model.layerset.mlp import MLP
from src.util.config import path_res_2d_pattern, path_res_numpy_array
from src.util.time import ymdt
from src.util.sequence import product


# TODO CNN フィルタ実装 1h
# TODO プレーン
# TODO スポーク

# TODO バッチ分割実装 15min
# TODO テストデータシャッフル 15min(18:35)
# TODO 誤りデータの表示と誤り例（深層学習　p）1h
# TODO Dropoutの実装 1h
# TODO モデルのサーフェスだけでなく、ソリッドも試す
# TODO EASY-CLASSIFIERの実装（クラス分類数を大まかなものに変更）30min

@client
def cubic_cnn(n_div=50, img_size=(256,256), is_boxel=False):
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

    train_ins = [inputs.flatten() for inputs in train_ins]
    test_ins = [inputs.flatten() for inputs in test_ins]

    print "train data : ", len(train_ins)
    print "test data : ", len(test_ins)
    print "train classes : ", len(set(train_ans))
    print "test classes : ", len(set(test_ans))

    """
    MODEL
    """

    print "preparing models..."

    n_in = n_div ** 3 if is_boxel else img_size

    # model = MLP(l1=ConvLayer2d(n_in, in_channel=1, out_channel=1, k_size=3))
    model = MLP(l1=Layer(product(n_in), 1000), l2=Layer(1000, 500))

    """
    TRAIN
    """
    train_accuracies = []
    test_accuracies = []

    n_iter = 100
    n_epoch = 10

    for i in range(n_iter):

        print "{}st learning...".format(i)

        train_accuracy = 0
        test_accuracy = 0

        # print model.l1.W.get_value()

        for j in range(n_epoch):
            train_accuracy += model.forward(inputs=train_ins,
                                            answers=train_ans)
            test_accuracy += model.forward(inputs=test_ins,
                                           answers=test_ans,
                                           updates=())

        train_accuracy /= n_epoch
        test_accuracy /= n_epoch

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
