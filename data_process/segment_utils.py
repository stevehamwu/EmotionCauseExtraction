#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/2 15:31
# @Author  : Steve Wu
# @Site    : 
# @File    : segment_utils.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import os
import random
import jieba
from pyltp import Segmentor
from pyhanlp import *
import thulac
import pkuseg


class Segment:
    def __init__(self):
        pass

    def cut(self, sentence):
        pass


class JiebaSegment(Segment):
    def __init__(self):
        super(JiebaSegment, self).__init__()
        self.segmentor = jieba

    def cut(self, sentence):
        return list(self.segmentor.cut(sentence))


class LtpSegment(Segment):
    def __init__(self):
        super(LtpSegment, self).__init__()
        LTP_DATA_DIR = os.path.join('/data10T/data/wujipeng/ltp_data_v3.4.0/')  # ltp模型目录的路径
        cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
        self.segmentor = Segmentor()
        self.segmentor.load(cws_model_path)

    def cut(self, sentence):
        return list(self.segmentor.segment(sentence))

    def __del__(self):
        self.segmentor.release()


class HanlpSegment(Segment):
    def __init__(self):
        super(HanlpSegment, self).__init__()
        self.segmentor = HanLP
        self.segmentor.segment('')

    def cut(self, sentence):
        return [term.word for term in self.segmentor.segment(sentence)]


class ThulacSegment(Segment):
    def __init__(self):
        super(ThulacSegment, self).__init__()
        self.segmentor = thulac.thulac(seg_only=True)  #默认模式

    def cut(self, sentence):
        return list(self.segmentor.cut(sentence, text=True))


class PkusegSegment(Segment):
    def __init__(self):
        super(PkusegSegment, self).__init__()
        self.segmentor = pkuseg.pkuseg()  # 以默认配置加载模型

    def cut(self, sentence):
        return list(self.segmentor.cut(sentence))
