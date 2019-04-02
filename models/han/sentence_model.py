#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/4 11:45
# @Author  : Steve Wu
# @Site    : 
# @File    : sentence_model.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import pickle
import torch
import torch.nn as nn
from .attention import BertSingleAttention


class SentenceWithPosition(nn.Module):
    def __init__(self, batch_size, word_rnn_size, rnn_size, rnn_layers, pos_size=103, pos_embedding_dim=300,
                 dropout=0.5, pos_embedding_file=None, fix_pos=True):
        super(SentenceWithPosition, self).__init__()
        self.batch_size = batch_size
        self.rnn_size = rnn_size
        self.pos_embedding = nn.Embedding(pos_size, pos_embedding_dim, padding_idx=0)
        if pos_embedding_file is not None:
            self.pos_embedding = self.pos_embedding.from_pretrained(
                torch.FloatTensor(pickle.load(open(pos_embedding_file, 'rb'))), freeze=fix_pos)
        if fix_pos:
            self.pos_embedding.requires_grad = False
        self.gru = nn.GRU(2 * word_rnn_size + pos_embedding_dim, self.rnn_size, num_layers=rnn_layers,
                          bidirectional=True, batch_first=True, dropout=dropout)
        self.attention = BertSingleAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, documents, poses):
        poses = self.pos_embedding(poses)
        doc_with_pos = torch.cat((documents, poses), dim=-1)
        outputs, _ = self.gru(doc_with_pos)
        return outputs, None


class SentenceAttention(nn.Module):
    def __init__(self, batch_size, word_rnn_size, rnn_size, rnn_layers, dropout=0.5):
        super(SentenceAttention, self).__init__()
        self.batch_size = batch_size
        self.rnn_size = rnn_size

        self.gru = nn.GRU(2 * word_rnn_size, self.rnn_size, num_layers=rnn_layers, bidirectional=True,
                          batch_first=True, dropout=dropout)
        self.attention = BertSingleAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, documents, poses):
        outputs, _ = self.gru(documents)
        values, sentence_attn = self.attention(outputs, outputs, outputs, dropout=self.dropout)
        return values, sentence_attn


class SentenceAttentionWithPosition(nn.Module):
    def __init__(self, batch_size, word_rnn_size, rnn_size, rnn_layers, pos_size=103, pos_embedding_dim=300,
                 dropout=0.5, pos_embedding_file=None, fix_pos=True):
        super(SentenceAttentionWithPosition, self).__init__()
        self.batch_size = batch_size
        self.rnn_size = rnn_size
        self.pos_embedding = nn.Embedding(pos_size, pos_embedding_dim, padding_idx=0)
        if pos_embedding_file is not None:
            self.pos_embedding = self.pos_embedding.from_pretrained(
                torch.FloatTensor(pickle.load(open(pos_embedding_file, 'rb'))), freeze=fix_pos)
        if fix_pos:
            self.pos_embedding.requires_grad = False
        self.gru = nn.GRU(2 * word_rnn_size + pos_embedding_dim, self.rnn_size, num_layers=rnn_layers,
                          bidirectional=True, batch_first=True, dropout=dropout)
        self.attention = BertSingleAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, documents, poses):
        poses = self.pos_embedding(poses)
        doc_with_pos = torch.cat((documents, poses), dim=-1)
        outputs, _ = self.gru(doc_with_pos)
        values, sentence_attn = self.attention(outputs, outputs, outputs, dropout=self.dropout)
        return values, sentence_attn


class SentenceAttentionWithPositionV2(nn.Module):
    def __init__(self, batch_size, word_rnn_size, rnn_size, rnn_layers, pos_size=103, pos_embedding_dim=300,
                 dropout=0.5, pos_embedding_file=None, fix_pos=True):
        super(SentenceAttentionWithPositionV2, self).__init__()
        self.batch_size = batch_size
        self.rnn_size = rnn_size
        self.pos_embedding = nn.Embedding(pos_size, pos_embedding_dim, padding_idx=0)
        if pos_embedding_file is not None:
            self.pos_embedding = self.pos_embedding.from_pretrained(
                torch.FloatTensor(pickle.load(open(pos_embedding_file, 'rb'))), freeze=fix_pos)
        if fix_pos:
            self.pos_embedding.requires_grad = False

        self.batch_norm = nn.BatchNorm1d(batch_size)
        self.gru = nn.GRU(2 * word_rnn_size + pos_embedding_dim, self.rnn_size, num_layers=rnn_layers,
                          bidirectional=True, batch_first=True, dropout=dropout)
        self.attention = BertSingleAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, documents, poses):
        documents = self.batch_norm(documents.transpose(0, 1)).transpose(0, 1)
        poses = self.pos_embedding(poses)
        doc_with_pos = torch.cat((documents, poses), dim=-1)
        outputs, _ = self.gru(doc_with_pos)
        values, sentence_attn = self.attention(outputs, outputs, outputs, dropout=self.dropout)
        return values, sentence_attn


def build_mask(poses):
    mask = torch.zeros_like(poses, dtype=torch.float32, requires_grad=False)
    indices = abs(poses - 69).argmin(1)
    for i, indice in enumerate(indices):
        mask[i, indice] = 1
    return mask


class SentenceAttentionWithPositionV3(nn.Module):
    def __init__(self, batch_size, word_rnn_size, rnn_size, rnn_layers, pos_size=103, pos_embedding_dim=300,
                 dropout=0.5, pos_embedding_file=None, fix_pos=True):
        super(SentenceAttentionWithPositionV3, self).__init__()
        self.batch_size = batch_size
        self.rnn_size = rnn_size
        self.pos_embedding = nn.Embedding(pos_size, pos_embedding_dim, padding_idx=0)
        if pos_embedding_file is not None:
            self.pos_embedding = self.pos_embedding.from_pretrained(
                torch.FloatTensor(pickle.load(open(pos_embedding_file, 'rb'))), freeze=fix_pos)
        if fix_pos:
            self.pos_embedding.requires_grad = False
        self.gru = nn.GRU(2 * word_rnn_size + pos_embedding_dim, self.rnn_size, num_layers=rnn_layers,
                          bidirectional=True, batch_first=True, dropout=dropout)
        self.attention = BertSingleAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, documents, poses):
        pos_mask = build_mask(poses)
        poses = self.pos_embedding(poses)

        doc_with_pos = torch.cat((documents, poses), dim=-1)
        outputs, _ = self.gru(doc_with_pos)

        # 用情感句做Attention
        pos_mask = pos_mask.unsqueeze(-1).expand_as(outputs)
        q = torch.mul(outputs, pos_mask).sum(1, keepdim=True)
        q = q.expand_as(outputs)

        values, sentence_attn = self.attention(q, outputs, outputs, dropout=self.dropout)
        return values, sentence_attn