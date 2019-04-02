#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/17 11:22
# @Author  : Steve Wu
# @Site    : 
# @File    : ecdataset.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import os
import numpy as np
from torch.utils.data import Dataset
from utils.data.process import load_data, pad_sequence


class ECDataset(Dataset):
    def __init__(self, data_root, train=True):
        super(ECDataset, self).__init__()
        self.train = train
        self.data_path = os.path.join(data_root, '{}_set.txt'.format('train' if train else 'dev'))
        self.read_data()
        self.read_vocab()

    def read_data(self):
        self.data = load_data(self.data_path)

    def read_vocab(self):
        with open('/data/wujipeng/ec/data/raw_data/vocab.txt', 'r') as f:
            self.word_unk = f.readline().strip()
            self.vocab = ['<pad>', self.word_unk] + f.readline().strip().split(' ')
        self.i2w = {i: w for i, w in enumerate(self.vocab)}
        self.w2i = {w: i for i, w in enumerate(self.vocab)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        n_id, n_clauses, n_keyword, n_emotion, n_pos, n_label = item
        n_clauses = [clause.strip().split(' ') for clause in n_clauses.split('\x01')]
        n_keyword = n_keyword.strip().split(' ')
        return tuple([n_id, n_clauses, n_keyword, n_emotion, n_pos, n_label])

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

    def collate_fn(self, batch_data, pad=True, clause_length=0, batch_size=16):
        batch_data = list(zip(*batch_data))
        ids, clauses, keywords, emotions, poses, labels = batch_data
        ids = [int(i) for i in ids]
        clauses = [self.word2idx(clause, batched=True) for clause in clauses]
        keywords = self.word2idx(keywords, batched=True)
        emotions = [int(e) for e in emotions]
        poses = [[int(p) for p in pos.strip().split(' ')] for pos in poses]
        labels = [[int(l) for l in label.strip().split(' ')] for label in labels]
        if pad:
            sentence_length = max([len(label) for label in labels])
            clauses = [pad_sequence(clause, clause_length) + [[0] * clause_length] * (sentence_length - len(clause)) for
                       clause in clauses]
            keywords = pad_sequence(keywords, clause_length)
            poses = [pose + [0] * (sentence_length - len(pose)) for pose in poses]
            labels = [label + [-100] * (sentence_length - len(label)) for label in labels]
            if len(ids) < batch_size:
                bs = batch_size - len(ids)
                ids += [0] * bs
                clauses += [[[0] * clause_length] * sentence_length] * bs
                keywords += [[0] * clause_length] * bs
                emotions += [0] * bs
                poses += [[0] * sentence_length] * bs
                labels += [[-100] * sentence_length] * bs

        return ids, np.array(clauses), np.array(keywords), np.array(emotions), np.array(poses), np.array(labels)

    @staticmethod
    def batch2input(batch):
        return batch[0], batch[1]

    @staticmethod
    def batch2target(batch):
        return batch[-1]