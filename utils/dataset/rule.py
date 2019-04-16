#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/22 9:49
# @Author  : Steve Wu
# @Site    :
# @File    : han.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.data.process import pad_sequence, load_data
import os
import pickle


class ECRuleDataset(Dataset):
    def __init__(self, data_root, vocab_root, batch_size=16, rule='all', train=True):
        super(ECRuleDataset, self).__init__()
        self.train = train
        self.data_path = os.path.join(data_root, '{}_set.txt'.format('train' if train else 'val'))
        self.vocab_root = vocab_root
        self.read_data()
        self.read_vocab()
        self.read_rule(rule if rule != 'all' else '_all')

    def read_data(self):
        self.data = load_data(self.data_path)

    def read_vocab(self):
        with open(os.path.join(self.vocab_root, 'vocab.txt'), 'r') as f:
            self.word_unk = f.readline().strip()
            self.vocab = ['<pad>', self.word_unk] + f.readline().strip().split(' ')
        self.i2w = {i: w for i, w in enumerate(self.vocab)}
        self.w2i = {w: i for i, w in enumerate(self.vocab)}

    def read_rule(self, rule):
        self.rules = pickle.load(open('/data10T/data/wujipeng/ec/data/rule/rules{}.pkl'.format(rule), 'rb'))
        self.rule_labels = pickle.load(open('/data10T/data/wujipeng/ec/data/rule/rule_labels{}.pkl'.format(rule), 'rb'))
        self.masks = pickle.load(open('/data10T/data/wujipeng/ec/data/rule/masks{}.pkl'.format(rule), 'rb'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        n_id, n_clauses, n_natures, n_keyword, n_emotion, n_pos, n_label = item
        n_clauses = [clause.strip().split(' ') for clause in n_clauses.split('\x01')]
        n_natures = [nature.strip().split(' ') for nature in n_natures.split('\x01')]
        n_keyword = n_keyword.strip().split(' ')
        n_pos = n_pos.strip().split(' ')
        return tuple([n_id, n_clauses, n_natures, n_keyword, n_emotion, n_pos, n_label])

    def word2idx(self, words, batched=False):
        if not batched:
            indices = [self.w2i[w] if w in self.w2i else self.w2i[self.word_unk] for w in words]
        else:
            indices = [[self.w2i[w] if w in self.w2i else self.w2i[self.word_unk] for w in item] for item in words]
        return indices

    def idx2word(self, indices, batched=False):
        if not batched:
            words = [self.i2w[i] for i in indices]
        else:
            words = [[self.i2w[i] for i in item] for item in indices]
        return words

    def collate_fn(self, batch_data, pad=True, clause_length=41, batch_size=16):
        batch_data = list(zip(*batch_data))
        ids, clauses, natures, keywords, emotions, poses, labels = batch_data
        ids = list(map(int, ids))
        clauses = [self.word2idx(clause, batched=True) for clause in clauses]
        keywords = self.word2idx(keywords, batched=True)
        masks = list(self.masks[np.array(ids) - 1])
        emotions = list(map(int, emotions))
        poses = [list(map(int, pos)) for pos in poses]
        labels = [[int(l) for l in label.strip().split(' ')] for label in labels]
        rules = list(self.rules[np.array(ids) - 1])
        rule_labels = list(self.rule_labels[np.array(ids) - 1])
        if pad:
            sentence_length = max([len(label) for label in labels])
            clauses = [pad_sequence(clause, clause_length) + [[0] * clause_length] * (sentence_length - len(clause)) for
                       clause in clauses]
            natures = [
                pad_sequence(nature, clause_length, pad='w') + [['w'] * clause_length] * (sentence_length - len(nature))
                for nature in natures]
            masks = [
                pad_sequence(list(map(list, mask)), clause_length, pad=0) + [[0] * clause_length] * (sentence_length - len(mask))
                for mask in masks]

            keywords = pad_sequence(keywords, clause_length)
            poses = [pos + [0] * (sentence_length - len(pos)) for pos in poses]
            labels = [label + [-100] * (sentence_length - len(label)) for label in labels]
            rule_labels = [list(label) + [-100] * (sentence_length - len(label)) for label in rule_labels]
            if len(ids) < batch_size:
                bs = batch_size - len(ids)
                ids += [0] * bs
                clauses += [[[0] * clause_length] * sentence_length] * bs
                natures += [[['w'] * clause_length] * sentence_length] * bs
                masks += [[[0] * clause_length] * sentence_length] * bs
                keywords += [[0] * clause_length] * bs
                emotions += [0] * bs
                poses += [[0] * sentence_length] * bs
                labels += [[-100] * sentence_length] * bs
                rules += [[0] * 15] * bs
                rule_labels += [-100] * bs

        return ids, np.array(clauses), natures, np.array(keywords), np.array(masks), np.array(emotions), np.array(
            poses), np.array(
            labels), np.array(rules), np.array(rule_labels)

    @staticmethod
    def batch2input(batch):
        return batch[1], batch[3], batch[4], batch[6]

    @staticmethod
    def batch2target(batch):
        return batch[-3]

    @staticmethod
    def batch2rule(batch):
        return batch[-2]

    @staticmethod
    def batch2rule_target(batch):
        return batch[-1]
