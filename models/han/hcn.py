#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/23 20:34
# @Author  : Steve Wu
# @Site    : 
# @File    : hcn.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/22 9:39
# @Author  : Steve Wu
# @Site    :
# @File    : han.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import torch
import torch.nn as nn
from .attention import BertSingleAttention
from .han import SentenceAttention


class WordCNN(nn.Module):
    def __init__(self,
                 embedding_dim,
                 batch_size,
                 sequence_length,
                 num_features,
                 kernel_sizes,
                 sentence_dim,
                 dropout=0.5):
        super(WordCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.kernel_sizes = kernel_sizes
        self.sentence_dim = sentence_dim

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_features, kernel_size=kernel_size),
            nn.BatchNorm1d(num_features=num_features),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size),
            nn.BatchNorm1d(num_features=num_features),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=sequence_length - kernel_size * 2 + 2)
        ) for kernel_size in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes)*num_features, sentence_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentences, keywords):
        x = sentences.view(-1, self.sequence_length, self.embedding_dim)
        x = x.transpose(1, 2)
        out = [conv(x).squeeze() for conv in self.convs]
        conv_out = torch.cat(out, dim=1)
        conv_out = conv_out.view(self.batch_size, -1, conv_out.size(-1))
        document = self.fc(self.dropout(conv_out))
        return document, None


class HierarchicalConvolutionNetwork(nn.Module):
    def __init__(self,
                 batch_size,
                 vocab_size,
                 num_classes,
                 embedding_dim,
                 sequence_length,
                 num_features,
                 kernel_sizes,
                 sentence_dim,
                 sentence_rnn_size,
                 sentence_rnn_layers,
                 sentence_att_size,
                 dropout=0.5,
                 fix_embed=True,
                 name='HAN'):
        super(HierarchicalConvolutionNetwork, self).__init__()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.sentence_rnn_size = sentence_rnn_size
        self.num_classes = num_classes
        self.fix_embed = fix_embed
        self.name = name

        self.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_cnn = WordCNN(
            embedding_dim,
            batch_size,
            sequence_length,
            num_features,
            kernel_sizes,
            sentence_dim,
            dropout=0.5
        )
        self.sentence_rnn = SentenceAttention(
            batch_size,
            sentence_dim // 2,
            sentence_rnn_size,
            sentence_rnn_layers,
            sentence_att_size,
            dropout=dropout)
        self.fc = nn.Linear(2 * sentence_rnn_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def init_weights(self, embeddings):
        if embeddings is not None:
            self.Embedding = self.Embedding.from_pretrained(
                embeddings, freeze=self.fix_embed)

    def forward(self, sentences, keywords):
        inputs = self.Embedding(sentences)
        queries = self.Embedding(keywords)
        documents, word_attn = self.word_cnn(inputs, queries)
        outputs, sentence_attn = self.sentence_rnn(documents)
        outputs = self.fc(self.dropout(outputs))
        outputs.view(-1, self.num_classes)
        return outputs, word_attn, sentence_attn
