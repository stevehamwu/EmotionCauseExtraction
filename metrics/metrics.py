#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/14 21:26
# @Author  : Steve Wu
# @Site    : 
# @File    : metrics.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import sys
from abc import ABCMeta, abstractmethod
from utils.app.log import Logger


class Metrics:
    __metaclass__ = ABCMeta

    def __init__(self, config):
        self.config = config
        # self.logger = Logger.get_instance()

    @abstractmethod
    def __call__(self, predicts, targets, probs):
        """
        """
        pass

    @abstractmethod
    def classification_report(self, predicts, targets):
        pass

