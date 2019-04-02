#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/8 17:21
# @Author  : Steve Wu
# @Site    : 
# @File    : config.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import json

from config.handlers import ConfigHandlerFactory


class Config:
    """
    config file format: yaml, json.
    """

    def __init__(self, conf = None):
        """
        conf: dict
        """
        self._state = conf

    # def default_config(self):
    #     default_file = os.path.join(os.path.dirname(__file__), "default.yaml")
    #     ConfigHandlerFactory(self).get_instance(default_file).load_config()

    def __iter__(self):
        return iter(self._state)

    def __len__(self):
        return len(self._state)

    def __setitem__(self, path, value, seperator="."):
        if seperator in path:
            path = path.split(seperator)
            conf = self._state
            for p in path[:-1]:
                if p in conf:
                    conf = conf[p]
                else:
                    conf[p] = {}
                    conf = conf[p]
            conf[path[-1]] = value
        else:
            self._state[path] = value

    def __getitem__(self, path, seperator="."):
        """
        Args
            path: str
        """
        return self.extract(path, seperator)

    def get(self, path, seperator="."):
        try:
            return self.extract(path, seperator)
        except KeyError as e:
            return None

    def extract(self, path, seperator="."):
        """
        Args:
            path: str
        """
        try:
            path = path.split(seperator)
            conf = self._state
            for p in path:
                conf = conf[p]
            if isinstance(conf, dict):
                return Config(conf)
            else:
                return conf
        except KeyError as e:
            raise KeyError('.'.join(path))

    def todict(self):
        return self._state

    def from_file(self, file_name):
        ConfigHandlerFactory(self).get_instance(file_name).load_config()

    def to_file(self, file_name):
        ConfigHandlerFactory(self).get_instance(file_name).save_config()

    def print_to(self, logger):
        js = json.dumps(self._state, indent=2)
        logger.info(js)