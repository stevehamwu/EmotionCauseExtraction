#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/13 20:53
# @Author  : Steve Wu
# @Site    : 
# @File    : metric.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
from .ft_rule import *
import numpy as np
import pandas as pd


def metrics(true_labels, pred_labels):
    succ, real_right, pred_right = 0, 0, 0
    for i in range(len(true_labels)):
        if true_labels[i] == 1:
            real_right += 1
            if pred_labels[i] == 1:
                succ += 1
        if pred_labels[i] == 1:
            pred_right += 1
    precision = succ / pred_right if pred_right > 0 else 0
    recall = succ / real_right
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def test_rules(data=pd.read_csv('/data/wujipeng/ec/data/han/processed_data.csv'), rule_range=range(1, 16), strict=True):
    pos0 = 69
    rules = np.zeros((len(data), 15))
    rule_labels = []
    for i, row in data.iterrows():
        clauses = [clause.strip().split(' ') for clause in row['clause'].split('\x01')]
        natures = [nature.strip().split(' ') for nature in row['nature'].split('\x01')]
        keyword = row['keyword'].replace(' ', '')
        emotion = row['emotion']
        clause_pos = [int(pos) - pos0 for pos in row['clause_pos'].split(' ')]
        rule_label = np.zeros(len(clauses), dtype=int)

        f = clause_pos.index(0)
        K = keyword

        for r in rule_range:
            if r in [6, 7, 8, 12, 13]:
                continue
            # constraints
            if r == 4:
                # 0,1: Happiness 2:Anger 3: Sadness 4: Fear 5:Disgust 6:Surprise
                if emotion not in [0, 1, 4, 6]:
                    continue
            if r in [14, 15]:
                if sum(rules[i]) > 0:
                    continue
            found = eval('rule{}'.format(r))(f, clauses, natures, K, strict)
            if found:
                rules[i][r - 1] = 1
                if found[0] == 'F':
                    rule_label[f] = 1
                elif found[0] == 'B':
                    rule_label[f - 1] = 1
                elif found[0] == 'A':
                    rule_label[f + 1] = 1
                break
        rule_labels.append(rule_label)
    return rule_labels
