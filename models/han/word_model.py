#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/4 11:44
# @Author  : Steve Wu
# @Site    : 
# @File    : word_model.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import pickle
import torch
import torch.nn as nn
from .attention import BertSingleAttention


class WordAttention(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 batch_size,
                 sequence_length,
                 rnn_size,
                 rnn_layers,
                 dropout=0.5):
        super(WordAttention, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers

        self.gru = nn.GRU(
            embedding_dim,
            rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout)
        self.attention = BertSingleAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentences, keywords):
        x = sentences.view(-1, self.sequence_length, self.embedding_dim)
        x, _ = self.gru(x)
        q, _ = self.gru(keywords)
        x = x.view(self.batch_size, -1, self.sequence_length, 2*self.rnn_size)
        q = q.unsqueeze(1).expand_as(x)

        mask = torch.matmul(keywords.unsqueeze(1).expand_as(sentences), sentences.transpose(-2, -1)) != 0
        values, word_attn = self.attention(q, x, x, mask=mask, dropout=self.dropout)
        values = values.sum(dim=2)
        word_attn = word_attn.sum(dim=2)

        return values, word_attn


class WordAttentionV2(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 batch_size,
                 sequence_length,
                 rnn_size,
                 rnn_layers,
                 dropout=0.5):
        super(WordAttentionV2, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers

        self.gru = nn.GRU(
            2 * embedding_dim,
            rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout)
        self.attention = BertSingleAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentences, keywords):
        k = keywords.unsqueeze(1).expand_as(sentences)
        max_k = k.max(2, keepdim=True)[0].expand_as(sentences)
        s_k = torch.cat((sentences, max_k), dim=-1)
        x = s_k.view(-1, self.sequence_length, 2*self.embedding_dim)
        x, _ = self.gru(x)
        values, word_attn = self.attention(x, x, x, dropout=self.dropout)
        values = values.sum(dim=1).view(self.batch_size, -1,
                                        2 * self.rnn_size)

        return values, word_attn


class WordAttentionWithPos(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 batch_size,
                 sequence_length,
                 rnn_size,
                 rnn_layers,
                 pos_size=103,
                 pos_embedding_dim=300,
                 pos_embedding_file=None,
                 fix_pos=True,
                 dropout=0.5):
        super(WordAttentionWithPos, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.pos_embedding_dim = pos_embedding_dim
        self.pos_embedding = nn.Embedding(pos_size, pos_embedding_dim, padding_idx=0)
        if pos_embedding_file is not None:
            self.pos_embedding = self.pos_embedding.from_pretrained(
                torch.FloatTensor(pickle.load(open(pos_embedding_file, 'rb'))), freeze=fix_pos)
        if fix_pos:
            self.pos_embedding.requires_grad = False

        self.gru = nn.GRU(
            embedding_dim,
            rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout)
        self.attention = BertSingleAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentences, keywords, poses):
        x = sentences.view(-1, self.sequence_length, self.embedding_dim)
        poses = self.pos_embedding(poses)
        x, _ = self.gru(x)
        q, _ = self.gru(keywords)
        q = q.unsqueeze(1).expand(self.batch_size, sentences.size(1),
                                  self.sequence_length, 2 * self.rnn_size)
        q = q.reshape_as(x)

        values, word_attn = self.attention(q, x, x, dropout=self.dropout)
        values = values.sum(dim=1).view(self.batch_size, -1,
                                        2 * self.rnn_size)
        outputs = torch.cat((values, poses), -1)
        return outputs, word_attn


class BiGRU(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 batch_size,
                 sequence_length,
                 rnn_size,
                 rnn_layers,
                 dropout=0.5):
        super(BiGRU, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers

        self.gru = nn.GRU(
            embedding_dim,
            rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentences, keywords):
        x = sentences.view(-1, self.sequence_length, self.embedding_dim)
        x, _ = self.gru(x)
        values = x.sum(dim=1).view(self.batch_size, -1, 2 * self.rnn_size)

        return values, None