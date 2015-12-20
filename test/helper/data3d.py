from src.helper.data3d import *
from src.helper.visualize import plot_3d, plot_voxel

__author__ = 'ren'


def test_psb(id=0):
    p = psb(id)
    plot_3d(p)


def test_psb_binvox(id=0):
    binvox = psb_binvox(id)
    plot_voxel(binvox)


def test_psb_binvoxs(ids=[1, 2, 3, 4, 5]):
    x_train, x_test, y_train, y_test = psb_binvoxs(ids)
    for x in x_train:
        plot_voxel(x)
    for x in x_test:
        plot_voxel(x)


def test_voxelize(id=0):
    p = psb(id)
    voxel = voxelize(p)
    plot_voxel(voxel)
