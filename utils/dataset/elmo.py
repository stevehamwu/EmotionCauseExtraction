#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/24 11:31
# @Author  : Steve Wu
# @Site    : 
# @File    : elmo.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
# no pad
import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from utils.data.process import load_data, pad_sequence


def elmo_mean(arr):
    return np.mean(arr, axis=0)


class ELMoECDataset(Dataset):
    def __init__(self, data_root, elmo_embed_file, kw_embed_file, train=True, collate_fn='collate_fn'):
        super(ELMoECDataset, self).__init__()
        self.train = train
        self.data_path = os.path.join(
            data_root, '{}_set.txt'.format('train' if train else 'val'))
        self.read_data()
        self.elmo_embedding = pickle.load(open(elmo_embed_file, 'rb'))
        self.kw_embedding = pickle.load(open(kw_embed_file, 'rb'))
        self.collate_fn = eval('self.' + collate_fn)

    def read_data(self):
        self.data = load_data(self.data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        n_id, n_clauses, n_natures, n_keyword, n_emotion, n_pos, n_label = item
        n_clauses = [
            clause.strip().split(' ') for clause in n_clauses.split('\x01')
        ]
        n_natures = [
            nature.strip().split(' ') for nature in n_natures.split('\x01')
        ]
        n_keyword = n_keyword.strip().split(' ')
        n_pos = n_pos.strip().split(' ')
        return tuple(
            [n_id, n_clauses, n_natures, n_keyword, n_emotion, n_pos, n_label])

    def avg_collate_fn(self,
                       batch_data,
                       clause_length=40,
                       pad=True,
                       batch_size=16):
        elmo_dim = 1024
        batch_data = list(zip(*batch_data))
        ids, sentences, natures, keywords, emotions, poses, labels = batch_data
        ids = list(map(int, ids))
        elmos = self.elmo_embedding[np.array(ids) - 1]
        elmos = [np.array(list(map(elmo_mean, elmo)), dtype=np.float32) for elmo in elmos]
        kw_elmos = self.kw_embedding[np.array(ids) - 1]
        keywords = np.array(list(map(elmo_mean, kw_elmos)))
        emotions = list(map(int, emotions))
        poses = [list(map(int, pos)) for pos in poses]
        labels = [[int(l) for l in label.strip().split(' ')]
                  for label in labels]
        if pad:
            sentence_size = max([len(sentence) for sentence in sentences])
            elmos = np.array(
                [
                    np.vstack(
                        (elmo, np.zeros(
                            (sentence_size - len(elmo), elmo_dim))))
                    for elmo in elmos
                ],
                dtype=np.float32)
            poses = [pos + [0] * (sentence_size - len(pos)) for pos in poses]
            labels = [
                label + [-100] * (sentence_size - len(label))
                for label in labels
            ]
            if len(ids) < batch_size:
                bs = batch_size - len(ids)
                ids += [0] * bs
                elmos = np.vstack((elmos,
                                   np.zeros((bs, sentence_size, elmo_dim), dtype=np.float32)))
                keywords = np.vstack((keywords, np.zeros((bs, elmo_dim))))
                emotions += [0] * bs
                poses += [[0] * sentence_size] * bs
                labels += [[-100] * sentence_size] * bs
        return ids, elmos, natures, keywords, emotions, np.array(
            poses), np.array(labels)

    def batch2input(self, batch):
        return batch[1], batch[3], batch[5]

    def batch2target(self, batch):
        return batch[-1]
