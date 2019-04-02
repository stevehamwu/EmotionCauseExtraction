#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/14 22:09
# @Author  : Steve Wu
# @Site    : 
# @File    : statistics.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
from abc import ABCMeta, abstractmethod
from utils.app.log import Logger


class Statistics:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.logger = Logger.get_instance()

    @abstractmethod
    def __call__(self, datasets, all_probs, model_path, epoch):
        """
        """
        pass