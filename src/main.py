# coding:utf-8

import sys
from src.client.image_recognition import image_recognition
from src.client.show_array import show_np_array

__author__ = 'ren'

if __name__ == '__main__':

    param = sys.argv
    client = param[1]

    if client == 'image_recognition':
        image_recognition()
    elif client == 'show_np_array':
        x_label = "iteration"
        y_label = "accuracy"
        y_lim = (0, 1)
        location = 'upper right'
        keyword = '11-23 13:15'
        show_np_array(x_label, y_label, y_lim, location=location,
                      keyword=keyword)

