#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/22 17:21
# @Author  : Steve Wu
# @Site    : 
# @File    : ec.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import itertools
import numpy as np
from metrics.metrics import Metrics
from utils.app.log import Logger
from sklearn.metrics import roc_auc_score


class ECMetrics(Metrics):

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
            if pred_labels[i] == true_labels[i]:
                if true_labels[i] == 0:
                    tn += 1
                else:
                    tp += 1
            else:
                if true_labels[i] == 0:
                    fp += 1
                else:
                    fn += 1
            all_pred.append(pred_labels[i])
            all_true.append(true_labels[i])
            all_probs.append(probs[i])
        acc = (tp + tn) / (tp + tn + fp + fn)
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0
        auc = roc_auc_score(all_true, all_probs) if sum(all_true) > 0 else 0.
        return tp + tn + fp + fn, acc, pre, rec, f1, auc
