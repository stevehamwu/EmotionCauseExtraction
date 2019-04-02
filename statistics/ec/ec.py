#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/14 22:10
# @Author  : Steve Wu
# @Site    : 
# @File    : ec.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import os
import numpy as np
from statistics.statistics import Statistics
from utils.app.log import Logger


class ECStatistics(Statistics):

    def __call__(self, dataset, all_probs, model_path, epoch):
        """
        Args:
            dataset: val_datasets
            targets: (bat, n(s)), 0-1
            preds: (bat, n(s)), 0-1
            probs: (bat, n(s), 2) 0-1
        """
        f = open(os.path.join(model_path, 'statistics_final_{}.csv'.format(epoch)), 'w')
        f.write(
            'SUCC\ttest_preds\ttest_labels\ttest_results[0]\ttest_results[1]\t'
            'sentence\tclause\tpredict\treal\tsucc\tkeyword\tcontent\n'
        )
        succ, pred_right, real_right = 0, 0, 0
        for i, data in enumerate(dataset):
            probs = np.array(all_probs[i])
            preds = probs.argmax(1)

            sid, clauses, natures, keyword, emotion, poses, labels = data
            labels = list(map(int, labels.split(' ')))
            max_prob = probs[:len(data[1]), 1].argmax()
            if labels[max_prob] == 1:
                succ += 1
            pred_right += 1
            real_right += labels.count(1)
            for j, clause in enumerate(clauses):
                SUC = 'SUC' if preds[j] == labels[j] and labels[j] == 1 else '###'
                label = labels[j]
                pred = preds[j]
                prob0, prob1 = probs[j]
                predict = 'c' if j == max_prob else 'n'
                real = 'c' if label == 1 else 'n'
                suc = 'suc' if j == max_prob and label == 1 else '###'
                f.write('{}\t{}\t{}\t{:.6f}\t{:.6f}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    SUC, pred, label, prob0, prob1, sid, j + 1, predict, real, suc,
                    ' '.join(keyword), ' '.join(clause)))
        self.logger.info('succ: {}'.format(succ))
        self.logger.info('predict_right: {}'.format(pred_right))
        self.logger.info('real_right: {}'.format(real_right))
        precision = float(succ) / pred_right
        recall = float(succ) / real_right
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1
