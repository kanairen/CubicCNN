from src.util.shape import *
from src.helper.data3d import psb_binvox
from src.helper.visualize import plot_boxel

__author__ = 'ren'


def test_rotate_voxel():
    boxel = psb_binvox(0)
    r_x = (45, 0, 0)
    r_y = (0, 45, 0)
    r_z = (0, 0, 45)
    plot_boxel(rotate_voxel(boxel, r_x))
    plot_boxel(rotate_voxel(boxel, r_y))
    plot_boxel(rotate_voxel(boxel, r_z))
