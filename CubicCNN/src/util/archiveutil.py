#!/usr/bin/env python
# coding: utf-8

import os
import zipfile


def unzip(from_path, to_path):
    with zipfile.ZipFile(from_path, 'r') as zip_file:
        if not os.path.exists(to_path):
            os.makedirs(to_path)
        for f in zip_file.namelist():
            with open(os.path.join(to_path, f), 'wb') as unzip_file:
                unzip_file.write(zip_file.read(f))
