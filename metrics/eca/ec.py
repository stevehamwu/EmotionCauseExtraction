#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/22 17:21
# @Author  : Steve Wu
# @Site    : 
# @File    : ec.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import numpy as np
from metrics.metrics import Metrics
from utils.app.log import Logger


class ECMetrics(Metrics):

    def __call__(self, pred_labels, true_labels):
        """
        Args:
            pred_labels: (bat, n(s)), 0-3
            true_labels: (bat, n(s)), 0-3
        """

        tp, tn, fp, fn = 0, 0, 0, 0
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
        acc = (tp + tn) / (tp + tn + fp + fn)
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0
        return tp + tn + fp + fn, acc, pre, rec, f1
