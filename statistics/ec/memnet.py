#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/21 10:45
# @Author  : Steve Wu
# @Site    : 
# @File    : memnet.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import os
import numpy as np
from statistics.statistics import Statistics
from utils.app.log import Logger


class MECStatistics(Statistics):

    def __call__(self, dataset, all_probs, model_path, epoch):
        """
        Args:
            dataset: val_datasets
            targets: (bat, n(s)), 0-1
            preds: (bat, n(s)), 0-1
            probs: (bat, n(s), 2) 0-1
        """
        probs = all_probs[:len(dataset)]
        preds = np.argmax(probs, axis=1)

        s = 0
        max_list = []
        for sid, cid, prob in zip(*list(zip(*dataset))[:2], probs):
            if sid != s:
                if s > 0:
                    max_list.append((s, max_cid))
                s = sid
                max_prob = 0
                max_cid = 0
            if prob[1] > max_prob:
                max_prob = prob[1]
                max_cid = cid
        max_list.append((sid, max_cid))

        f = open(os.path.join(model_path, 'statistics_final_{}.csv'.format(epoch)), 'w')
        f.write(
            'SUCC\ttest_preds\ttest_labels\ttest_results[0]\ttest_results[1]\t'
            'sentence\tclause\tpredict\treal\tsucc\tkeyword\tcontent\n'
        )
        succ, pred_right, real_right = 0, 0, 0
        for i, data in enumerate(dataset):
            sid, cid, clause, nature, keyword, emotion, pos, label = data
            pred = preds[i]
            prob0, prob1 = probs[i]
            SUC = 'SUC' if pred == label and label == 1 else '###'
            predict = 'c' if (sid, cid) in max_list else 'n'
            real = 'c' if label == 1 else 'n'
            suc = 'suc' if predict == 'c' and label == 1 else '###'
            f.write('{}\t{}\t{}\t{:.6f}\t{:.6f}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                SUC, pred, label, prob0, prob1, sid, cid, predict, real, suc,
                ' '.join(keyword), ' '.join(clause)))

            if (sid, cid) in max_list:
                pred_right += 1
                if label == 1:
                    succ += 1
            if label == 1:
                real_right += 1

        self.logger.info('succ: {}'.format(succ))
        self.logger.info('predict_right: {}'.format(pred_right))
        self.logger.info('real_right: {}'.format(real_right))
        precision = float(succ) / pred_right
        recall = float(succ) / real_right
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.
        return precision, recall, f1
