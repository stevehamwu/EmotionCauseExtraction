#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/22 9:39
# @Author  : Steve Wu
# @Site    : 
# @File    : HierarchicalAttentionNetwork.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
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
                 att_size,
                 dropout=0.5,
                 embeddings=None):
        super(WordAttention, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.att_size = att_size

        self.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embeddings is not None:
            self.Embedding = self.Embedding.from_pretrained(
                torch.FloatTensor(embeddings))

        self.gru = nn.GRU(
            embedding_dim,
            rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True)
        self.attention = BertSingleAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentences, keywords):
        input_x = self.Embedding(sentences)
        input_q = self.Embedding(keywords)

        x = input_x.view(-1, self.sequence_length, self.embedding_dim)
        x, _ = self.gru(x)
        q, _ = self.gru(input_q)
        q = q.unsqueeze(1).expand(self.batch_size, input_x.size(1),
                                  self.sequence_length, 2 * self.rnn_size)
        q = q.reshape_as(x)

        values, word_attn = self.attention(q, x, x, dropout=self.dropout)
        values = values.sum(dim=1).view(self.batch_size, -1,
                                        2 * self.rnn_size)

        return values, word_attn


class SentenceAttention(nn.Module):
    def __init__(self, batch_size, word_rnn_size, sentence_rnn_size, sentence_rnn_layers, sentence_att_size,
                 dropout=0.5):
        super(SentenceAttention, self).__init__()
        self.batch_size = batch_size
        self.rnn_size = sentence_rnn_size

        self.gru = nn.GRU(2 * word_rnn_size, self.rnn_size, num_layers=sentence_rnn_layers, bidirectional=True,
                          dropout=dropout, batch_first=True)
        self.attention = BertSingleAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, documents):
        outputs, _ = self.gru(documents)
        values, sentence_attn = self.attention(outputs, outputs, outputs, dropout=self.dropout)
        return values, sentence_attn


class HierachicalAttentionNetwork(nn.Module):
    def __init__(self,
                 batch_size,
                 vocab_size,
                 embedding_dim,
                 sequence_length,
                 word_rnn_size,
                 word_rnn_layers,
                 word_att_size,
                 sentence_rnn_size,
                 sentence_rnn_layers,
                 sentence_att_size,
                 num_classes,
                 dropout=0.5,
                 embeddings=None):
        super(HierachicalAttentionNetwork, self).__init__()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.word_rnn_size = word_rnn_size
        self.sentence_rnn_size = sentence_rnn_size
        self.embeddings = embeddings

        self.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_rnn = WordAttention(
            vocab_size,
            embedding_dim,
            batch_size,
            sequence_length,
            word_rnn_size,
            word_rnn_layers,
            word_att_size,
            dropout=dropout,
            embeddings=embeddings)
        self.sentence_rnn = SentenceAttention(
            batch_size,
            word_rnn_size,
            sentence_rnn_size,
            sentence_rnn_layers,
            sentence_att_size,
            dropout=dropout)
        self.fc = nn.Linear(2 * sentence_rnn_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        if self.embeddings is not None:
            self.Embedding = self.Embedding.from_pretrained(
                torch.FloatTensor(self.embeddings))

    def forward(self, sentences, keywords):
        inputs = sentences
        queries = keywords
        documents, word_attn = self.word_rnn(inputs, queries)
        outputs, sentence_attn = self.sentence_rnn(documents)
        outputs = self.fc(self.dropout(outputs))
        return outputs, word_attn, sentence_attn

