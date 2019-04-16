#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/3 16:31
# @Author  : Steve Wu
# @Site    : 
# @File    : load_corpus.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
from segment_utils import *


def build_corpus(mode='hanlp'):
    with open('/data10T/data/wujipeng/ec/raw_data/corpus.txt') as f:
        raw_corpus = f.readlines()
    if mode == 'hanlp':
        segmentor = HanlpSegment()
    elif mode == 'jieba':
        segmentor = JiebaSegment()
    elif mode == 'ltp':
        segmentor = LtpSegment()
    elif mode == 'thulac':
        segmentor = ThulacSegment()
    elif mode == 'pku':
        segmentor = PkusegSegment()
    corpus = []
    for line in raw_corpus:
        for clause in line.strip().replace(' ', '').split('\x01'):
            corpus.append(segmentor.cut(clause))
    return corpus


if __name__ == '__main__':
    tools = ['hanlp', 'jieba', 'ltp', 'thulac', 'pku']
    for tool in tools:
        corpus = build_corpus(tool)
        with open('/data10T/data/wujipeng/ec/data/embedding/{}_corpus.txt'.format(tool), 'w') as f:
            f.write('\n'.join([' '.join(clause) for clause in corpus]))
