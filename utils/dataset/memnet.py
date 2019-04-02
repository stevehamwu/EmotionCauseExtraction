#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/19 23:11
# @Author  : Steve Wu
# @Site    : 
# @File    : memnet.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.data.process import load_data, pad_sequence, pad_memory
import os


class MECDataset(Dataset):
    def __init__(self, data_root, vocab_root, batch_size=16, train=True):
        super(MECDataset, self).__init__()
        self.train = train
        self.data_path = os.path.join(data_root, '{}_set.txt'.format('train' if train else 'val'))
        self.vocab_root = vocab_root
        self.batch_size = batch_size
        self.data = []
        self.read_data()
        self.read_vocab()
        self.read_pos()

    def read_data(self):
        data = load_data(self.data_path)
        for item in data:
            n_id, n_clauses, n_natures, n_keyword, n_emotion, n_pos, n_label = item
            n_id = int(n_id)
            n_clauses = [clause.strip().split(' ') for clause in n_clauses.split('\x01')]
            n_natures = [nature.strip().split(' ') for nature in n_natures.split('\x01')]
            n_keyword = n_keyword.replace(' ', '')
            n_pos = list(map(int, n_pos.strip().split(' ')))
            n_label = list(map(int, n_label.strip().split(' ')))
            for cid, (clause, nature, pos, label) in enumerate(zip(n_clauses, n_natures, n_pos, n_label)):
                self.data.append(tuple([n_id, cid + 1, clause, nature, n_keyword, n_emotion, pos, label]))

    def read_vocab(self):
        if os.path.isdir(self.vocab_root):
            self.vocab_root = os.path.join(self.vocab_root, 'vocab.txt')
        with open(self.vocab_root, 'r') as f:
            self.word_unk = f.readline().strip()
            self.vocab = ['<pad>', self.word_unk] + f.readline().strip().split(' ')
        self.i2w = {i: w for i, w in enumerate(self.vocab)}
        self.w2i = {w: i for i, w in enumerate(self.vocab)}

    def read_pos(self):
        pos_label = [
            'AAAA', 'AAAB', 'AAAC', 'AAAD', 'AABA', 'AABB', 'AABC', 'AABD', 'AACA',
            'AACB', 'AACC', 'AACD', 'AADA', 'AADB', 'AADC', 'AADD', 'ABAA', 'ABAB',
            'ABAC', 'ABAD', 'ABBA', 'ABBB', 'ABBC', 'ABBD', 'ABCA', 'ABCB', 'ABCC',
            'ABCD', 'ABDA', 'ABDB', 'ABDC', 'ABDD', 'ACAA', 'ACAB', 'ACAC', 'ACAD',
            'ACBA', 'ACBB', 'ACBC', 'ACBD', 'ACCA', 'ACCB', 'ACCC', 'ACCD', 'ACDA',
            'ACDB', 'ACDC', 'ACDD', 'ADAA', 'ADAB', 'ADAC', 'ADAD', 'ADBA', 'ADBB',
            'ADBC', 'ADBD', 'ADCA', 'ADCB', 'ADCC', 'ADCD', 'ADDA', 'ADDB', 'ADDC',
            'ADDD', 'BAAA', 'BAAB', 'BAAC', 'BAAD', 'BABA', 'BABB', 'BABC', 'BABD',
            'BACA', 'BACB', 'BACC', 'BACD', 'BADA', 'BADB', 'BADC', 'BADD', 'BBAA',
            'BBAB', 'BBAC', 'BBAD', 'BBBA', 'BBBB', 'BBBC', 'BBBD', 'BBCA', 'BBCB',
            'BBCC', 'BBCD', 'BCAA', 'BCAB', 'BCAC', 'BCAD', 'BDAA', 'BDAB', 'BDAC',
            'BDAD', 'BCBA', 'BCBB', 'BCBC'
        ]
        self.p2l = {i + 1: w for i, w in enumerate(pos_label)}
        self.l2p = {w: i + 1 for i, w in enumerate(pos_label)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

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

    def pos2label(self, poses, batched=False):
        if not batched:
            indices = [self.p2l[p] for p in poses]
        else:
            indices = [[self.p2l[p] for p in pos] for pos in poses]
        return indices

    def collate_fn(self, batch_data, pad=True, memory_size=41, sequence_size=3, batch_size=16):
        batch_data = list(zip(*batch_data))
        ids, cids, clauses, natures, keywords, emotions, poses, labels = batch_data
        ids = list(ids)
        cids = list(cids)
        clauses = self.word2idx(clauses, batched=True)
        keywords = self.word2idx(keywords, batched=False)
        emotions = list(map(int, emotions))
        poses = self.word2idx(self.pos2label(poses, batched=False))
        labels = list(labels)
        if pad:
            clauses = pad_memory(clauses, poses, memory_size, sequence_size, pad=0)
            keywords = [[keyword] * sequence_size for keyword in keywords]
            if len(ids) < batch_size:
                bs = batch_size - len(ids)
                ids += [0] * bs
                cids += [0] * bs
                clauses += [[[0] * sequence_size] * memory_size] * bs
                keywords += [[0] * sequence_size] * bs
                emotions += [0] * bs
                labels += [-100] * bs

        return ids, cids, np.array(clauses), natures, np.array(keywords), np.array(emotions), poses, np.array(
            labels)

    @staticmethod
    def batch2input(batch):
        return batch[2], batch[4]

    @staticmethod
    def batch2target(batch):
        return batch[-1]


# class MECDataset(Dataset):
#     def __init__(self, data_root, vocab_root, batch_size=16, train=True):
#         super(MECDataset, self).__init__()
#         self.train = train
#         self.data_path = os.path.join(data_root, '{}_set.txt'.format('train' if train else 'val'))
#         self.vocab_root = vocab_root
#         self.batch_size = batch_size
#         self.data = self.read_data()
#         self.read_vocab()
#         self.read_pos()
#
#     def read_data(self):
#         return load_data(self.data_path)
#
#     def read_vocab(self):
#         if os.path.isdir(self.vocab_root):
#             self.vocab_root = os.path.join(self.vocab_root, 'vocab.txt')
#         with open(self.vocab_root, 'r') as f:
#             self.word_unk = f.readline().strip()
#             self.vocab = ['<pad>', self.word_unk] + f.readline().strip().split(' ')
#         self.i2w = {i: w for i, w in enumerate(self.vocab)}
#         self.w2i = {w: i for i, w in enumerate(self.vocab)}
#
#     def read_pos(self):
#         pos_label = [
#             'AAAA', 'AAAB', 'AAAC', 'AAAD', 'AABA', 'AABB', 'AABC', 'AABD', 'AACA',
#             'AACB', 'AACC', 'AACD', 'AADA', 'AADB', 'AADC', 'AADD', 'ABAA', 'ABAB',
#             'ABAC', 'ABAD', 'ABBA', 'ABBB', 'ABBC', 'ABBD', 'ABCA', 'ABCB', 'ABCC',
#             'ABCD', 'ABDA', 'ABDB', 'ABDC', 'ABDD', 'ACAA', 'ACAB', 'ACAC', 'ACAD',
#             'ACBA', 'ACBB', 'ACBC', 'ACBD', 'ACCA', 'ACCB', 'ACCC', 'ACCD', 'ACDA',
#             'ACDB', 'ACDC', 'ACDD', 'ADAA', 'ADAB', 'ADAC', 'ADAD', 'ADBA', 'ADBB',
#             'ADBC', 'ADBD', 'ADCA', 'ADCB', 'ADCC', 'ADCD', 'ADDA', 'ADDB', 'ADDC',
#             'ADDD', 'BAAA', 'BAAB', 'BAAC', 'BAAD', 'BABA', 'BABB', 'BABC', 'BABD',
#             'BACA', 'BACB', 'BACC', 'BACD', 'BADA', 'BADB', 'BADC', 'BADD', 'BBAA',
#             'BBAB', 'BBAC', 'BBAD', 'BBBA', 'BBBB', 'BBBC', 'BBBD', 'BBCA', 'BBCB',
#             'BBCC', 'BBCD', 'BCAA', 'BCAB', 'BCAC', 'BCAD', 'BDAA', 'BDAB', 'BDAC',
#             'BDAD', 'BCBA', 'BCBB', 'BCBC'
#         ]
#         self.p2l = {i + 1: w for i, w in enumerate(pos_label)}
#         self.l2p = {w: i + 1 for i, w in enumerate(pos_label)}
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         item = self.data[index]
#         n_id, n_clauses, n_natures, n_keyword, n_emotion, n_pos, n_label = item
#         n_clauses = [clause.strip().split(' ') for clause in n_clauses.split('\x01')]
#         n_natures = [nature.strip().split(' ') for nature in n_natures.split('\x01')]
#         n_keyword = n_keyword.replace(' ', '')
#         n_pos = n_pos.strip().split(' ')
#         return tuple([n_id, n_clauses, n_natures, n_keyword, n_emotion, n_pos, n_label])
#
#     def word2idx(self, words, batched=False):
#         if not batched:
#             indices = [self.w2i[w] if w in self.w2i else self.w2i[self.word_unk] for w in words]
#         else:
#             indices = [[self.w2i[w] if w in self.w2i else self.w2i[self.word_unk] for w in item] for item in words]
#         return indices
#
#     def idx2word(self, indices, batched=False):
#         if not batched:
#             words = [self.i2w[i] for i in indices]
#         else:
#             words = [[self.i2w[i] for i in item] for item in indices]
#         return words
#
#     def pos2label(self, poses, batched=False):
#         if not batched:
#             indices = [self.p2l[p] for p in poses]
#         else:
#             indices = [[self.p2l[p] for p in pos] for pos in poses]
#         return indices
#
#     def collate_fn(self, batch_data, pad=True, memory_size=41, sequence_size=3, batch_size=16):
#         batch_data = list(zip(*batch_data))
#         ids, clauses, natures, keywords, emotions, poses, labels = batch_data
#         ids = list(map(int, ids))
#         clauses = [self.word2idx(clause, batched=True) for clause in clauses]
#         keywords = self.word2idx(keywords, batched=False)
#         emotions = list(map(int, emotions))
#         poses = [self.word2idx(self.pos2label(list(map(int, pos)))) for pos in poses]
#         labels = [[int(l) for l in label.strip().split(' ')] for label in labels]
#         if pad:
#             sentence_length = max([len(label) for label in labels])
#             clauses = [
#                 pad_memory(clause, pos, memory_size, sequence_size, pad=0) + [[[0] * sequence_size] * memory_size] * (
#                         sentence_length - len(clause)) for clause, pos in zip(clauses, poses)]
#             natures = [pad_memory(nature, pos, memory_size, sequence_size, pad='w') + [
#                 [['w'] * sequence_size] * memory_size] * (sentence_length - len(nature)) for nature, pos in
#                        zip(natures, poses)]
#             keywords = [[keyword] * sequence_size for keyword in keywords]
#             labels = [label + [-100] * (sentence_length - len(label)) for label in labels]
#             if len(ids) < batch_size:
#                 bs = batch_size - len(ids)
#                 ids += [0] * bs
#                 clauses += [[[[0] * sequence_size] * memory_size] * sentence_length] * bs
#                 keywords += [[0] * sequence_size] * bs
#                 emotions += [0] * bs
#                 labels += [[-100] * sentence_length] * bs
#
#         return ids, np.array(clauses), natures, np.array(keywords), np.array(emotions), poses, np.array(
#             labels)
#
#     @staticmethod
#     def batch2input(batch):
#         return batch[1], batch[3]
#
#     @staticmethod
#     def batch2target(batch):
#         return batch[-1]
