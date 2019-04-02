#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/21 8:27
# @Author  : Steve Wu
# @Site    : 
# @File    : memnet.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import itertools
import numpy as np
from metrics.metrics import Metrics
from utils.app.log import Logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class MECMetrics(Metrics):

    def __call__(self, pred_labels, true_labels, probs):
        """
        Args:
            pred_labels: (bat, n(s)), 0-3
            true_labels: (bat, n(s)), 0-3
        """
        if type(pred_labels[0]) != int:
            pred_labels = list(itertools.chain.from_iterable(pred_labels))
            true_labels = list(itertools.chain.from_iterable(true_labels))

        tp, tn, fp, fn = 0, 0, 0, 0
        all_pred, all_true, all_probs = [], [], []
        for i in range(len(pred_labels)):
            if true_labels[i] == self.config['ignore_index']:
                continue
            all_pred.append(pred_labels[i])
            all_true.append(true_labels[i])
            all_probs.append(probs[i])
        acc = accuracy_score(all_true, all_pred)
        pre = precision_score(all_true, all_pred)
        rec = recall_score(all_true, all_pred)
        f1 = f1_score(all_true, all_pred)
        auc = roc_auc_score(all_true, all_probs) if sum(all_true) > 0 else 0.
        return tp + tn + fp + fn, acc, pre, rec, f1, auc