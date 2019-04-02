#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/8 17:06
# @Author  : Steve Wu
# @Site    : 
# @File    : log.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu

import os
import logging
import logging.config


class Logger:
    __logger = None
    __format = None

    @staticmethod
    def init_instance(log_file=None, name=None, format=None):
        global __logger
        # logger
        __logger = logging.getLogger(name)
        __logger.setLevel(logging.INFO)

        # formatter
        global __format
        __format = (format if format is not None
                    else "%(asctime)s - %(filename)s - %(levelname)s - %(message)s")
        formatter = logging.Formatter(__format)

        # handlers
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        __logger.addHandler(sh)

        if log_file is not None:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            __logger.addHandler(fh)

    @staticmethod
    def get_instance():
        return __logger


if __name__ == '__main__':
    logger = Logger()
