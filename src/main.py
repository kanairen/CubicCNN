# coding:utf-8

import sys
from src.client import image_recognition, solid_recognition
from src.client.show_array import show_np_array

__author__ = 'ren'

if __name__ == '__main__':

    param = sys.argv
    client = param[1]

    if client == 'image_recognition':
        image_recognition.image_recognition()
    elif client == 'solid_recognition':
        data_type = param[2]
        solid_recognition.solid_recognition(data_type=data_type)
    elif client == 'show_np_array':
        x_label = "iteration"
        y_label = "accuracy"
        y_lim = (0, 1)
        location = 'upper left'
        keyword = '11-25'
        show_np_array(x_label, y_label, y_lim, location=location,
                      keyword=keyword, save=True)
