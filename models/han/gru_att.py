#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/5 19:43
# @Author  : Steve Wu
# @Site    : 
# @File    : gru_att.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import pickle

import torch
import torch.nn as nn
from .attention import BertSingleAttention
from .word_model import *
from .sentence_model import *


class GRUAttention(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 embedding_dim,
                 word_model,
                 dropout=0.5,
                 fix_embed=True,
                 name='HAN'):
        super(GRUAttention, self).__init__()
        self.num_classes = num_classes
        self.fix_embed = fix_embed
        self.name = name

        self.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_rnn = eval(word_model['class'])(**word_model['args'])
        self.fc = nn.Linear(2 * word_model['args']['rnn_size'] + word_model['args']['pos_embedding_dim'], num_classes)
        self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Sequential(
        #     nn.Linear(2 * self.sentence_rnn_size, linear_hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(linear_hidden_dim, num_classes)
        # )

    def init_weights(self, embeddings):
        if embeddings is not None:
            self.Embedding = self.Embedding.from_pretrained(
                embeddings, freeze=self.fix_embed)

    def forward(self, sentences, keywords, poses):
        inputs = self.Embedding(sentences)
        queries = self.Embedding(keywords)
        documents, word_attn = self.word_rnn(inputs, queries, poses)
        outputs = self.fc(self.dropout(documents))
        return outputs, word_attn, None
