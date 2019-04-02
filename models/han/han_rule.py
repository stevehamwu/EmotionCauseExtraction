#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/26 10:33
# @Author  : Steve Wu
# @Site    : 
# @File    : han_rule.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import pickle
import numpy as np
import torch
import torch.nn as nn
from .attention import *
from .word_model import *
from .sentence_model import *


class HierarchicalAttentionNetworkV3Rule(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 rule_classes,
                 embedding_dim,
                 word_model,
                 sentence_model,
                 dropout=0.5,
                 fix_embed=True,
                 name='HAN'):
        super(HierarchicalAttentionNetworkV3Rule, self).__init__()
        self.word_rnn_size = word_model['args']['rnn_size']
        self.sentence_rnn_size = sentence_model['args']['rnn_size']
        self.num_classes = num_classes
        self.rule_classes = rule_classes
        self.fix_embed = fix_embed
        self.name = name

        self.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_rnn = eval(word_model['class'])(**word_model['args'])
        self.sentence_rnn = eval(sentence_model['class'])(**sentence_model['args'])
        self.fc1 = nn.Linear(2 * self.word_rnn_size + 2 * self.sentence_rnn_size, num_classes)
        self.dropout1 = nn.Dropout(dropout)
        self.attention = NonLinearAttention(2 * self.sentence_rnn_size)
        self.fc2 = nn.Linear(2 * self.word_rnn_size + 2 * self.sentence_rnn_size, num_classes)
        self.dropout2 = nn.Dropout(dropout)
        # self.fc = nn.Sequential(
        #     nn.Linear(2 * self.sentence_rnn_size, linear_hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(linear_hidden_dim, num_classes)
        # )

    def init_weights(self, embeddings):
        if embeddings is not None:
            if isinstance(embeddings, np.ndarray):
                embeddings = torch.from_numpy(embeddings.astype(np.float32))
            self.Embedding.weight.data.copy_(embeddings)
            self.Embedding.weight.requires_grad = not self.fix_embed
            self.Embedding.padding_idx = 0

            # self.Embedding = self.Embedding.from_pretrained(embeddings, padding_idx=0, freeze=self.fix_embed)

    def forward(self, sentences, keywords, poses):
        inputs = self.Embedding(sentences)
        queries = self.Embedding(keywords)
        documents, word_attn = self.word_rnn(inputs, queries)
        outputs, sentence_attn = self.sentence_rnn(documents, poses)
        # outputs = self.fc(outputs)
        s_c = torch.cat((documents, outputs), dim=-1)
        outputs = self.fc1(self.dropout1(s_c))
        rule_outputs, sentence_attn = self.attention(s_c)
        rule_outputs = self.fc2(self.dropout2(rule_outputs))
        return rule_outputs, outputs, word_attn, sentence_attn
