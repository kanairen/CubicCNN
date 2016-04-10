#!/usr/bin/env python
# coding: utf-8

import os
import zipfile


def unzip(from_path, to_path):
    with zipfile.ZipFile(from_path, 'r') as zip_file:
        if not os.path.exists(to_path):
            os.makedirs(to_path)
        for member_name in zip_file.namelist():
            if ".." in member_name or member_name[0] == "/":
                raise IOError
        zip_file.extractall(to_path)
