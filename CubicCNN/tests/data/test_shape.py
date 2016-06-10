#!/usr/bin/env python
# coding: utf-8
from CubicCNN.src.data.shape import SHRECVoxel
from CubicCNN.src.util.plotutil import plot_voxel
from CubicCNN.src.util.parseutil import parse_cla
from CubicCNN import PATH_RES_SHAPE_SHREC_CLASS_TEST_CLA


def test_shrec_voxel():
    """
    unittestを使わないSHRECVoxelの簡易テスト
    :return:
    """
    # TODO:SHRECデータセットをダウンロード・binvox変換の機構ができたら適宜unittest対応
    classes = parse_cla(PATH_RES_SHAPE_SHREC_CLASS_TEST_CLA).keys()
    shrec = SHRECVoxel.create(n_fold=6)
    for x, y in zip(shrec.x_train, shrec.y_train):
        print classes[y]
        plot_voxel(x[0])
