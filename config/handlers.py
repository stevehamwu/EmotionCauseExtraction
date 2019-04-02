#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/8 17:21
# @Author  : Steve Wu
# @Site    : 
# @File    : handlers.py.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import os
import sys

import json
import yaml

sys.path.append("..")
from abc import ABCMeta, abstractmethod


class ConfigHandlerFactory(object):

    def __init__(self, config):
        self.config = config

    def get_instance(self, file_name):
        ext = os.path.splitext(file_name)[1]
        if ext == '.yaml':
            return YAMLConfigHandler(self.config, file_name)
        elif ext == 'json':
            return JSONConfigHandler(self.config, file_name)
        else:
            raise NotImplementedError("Unsupported type `{}`".format(ext))


class ConfigHandler(object):
    __metaclass__ = ABCMeta

    def __init__(self, config, file_name):
        self.config = config
        self.file_name = file_name

    @abstractmethod
    def load_config(self):
        pass

    @abstractmethod
    def save_config(self):
        pass


class YAMLConfigHandler(ConfigHandler):

    def load_config(self):
        with open(self.file_name, "r", encoding="utf8") as fd:
            # update config with data
            conf = yaml.safe_load(fd.read())
            self.config._state = conf

    def save_config(self):
        with open(self.file_name, "w", encoding="utf8") as fd:
            # transform config to data
            yaml.dump(self.config._state, fd)


class JSONConfigHandler(ConfigHandler):

    def load_config(self):
        with open(self.file_name, "r", encoding="utf8") as fd:
            # update config with data
            conf = json.load(fd)
            self.config._state = conf

    def save_config(self):
        with open(self.file_name, "w", encoding="utf8") as fd:
            # transform config to data
            js = json.dumps(self.config._state, indent=2)
