#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/20 16:25
# @Author  : Steve Wu
# @Site    : 
# @File    : memnet.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def position_encoding(sentence_size, embedding_size):
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)


class MemN2N(nn.Module):
    def __init__(self,
                 batch_size,
                 vocab_size,
                 embedding_dim,
                 sentence_size,
                 memory_size,
                 hops,
                 num_classes,
                 dropout=0.5,
                 fix_embed=True,
                 name='MemN2N'):
        super(MemN2N, self).__init__()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.sentence_size = sentence_size
        self.memory_size = memory_size
        self.hops = hops
        self.num_classes = num_classes
        self.fix_embed = fix_embed
        self.name = name

        self.encoding = torch.from_numpy(position_encoding(1, sentence_size * embedding_dim))
        self.Embedding = nn.Embedding(vocab_size, embedding_dim)

        self.fc = nn.Linear(3 * embedding_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def init_weights(self, embeddings):
        if embeddings is not None:
            self.Embedding.weight.data.copy_(embeddings)
            self.Embedding.weight.requires_grad = not self.fix_embed

    def set_device(self, device):
        self.encoding = self.encoding.to(device)

    def forward(self, stories, queries):
        q_emb0 = self.Embedding(queries)
        q_emb = q_emb0.view(-1, 1, 3 * self.embedding_dim)
        u_0 = torch.sum(q_emb * self.encoding, 1)
        u = [u_0]

        for i in range(self.hops):
            m_emb0 = self.Embedding(stories)
            m_emb = m_emb0.view(-1, self.memory_size, 1, 3 * self.embedding_dim)
            m = torch.sum(m_emb * self.encoding, -2)

            u_temp = u[-1].unsqueeze(-1).transpose(-2, -1)
            dotted = torch.sum(m * u_temp, -1)
            probs = F.softmax(dotted, -1)
            probs_temp = probs.unsqueeze(-1).transpose(-2, -1)

            c_emb0 = self.Embedding(stories)
            c_emb = c_emb0.view(-1, self.memory_size, 1, 3 * self.embedding_dim)
            c_temp = torch.sum(c_emb * self.encoding, -2)
            c = c_temp.transpose(-2, -1)

            o_k = torch.sum(c * probs_temp, -1)
            u_k = u[-1] + o_k
            u.append(u_k)

        outputs = self.dropout(self.fc(u_k))
        return outputs

    def gradient_noise_and_clip(self, parameters, device,
                                 noise_stddev=1e-3, max_clip=40.0):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        norm = nn.utils.clip_grad_norm_(parameters, max_clip)

        for p in parameters:
            noise = torch.randn(p.size()) * noise_stddev
            noise = noise.to(device)
            p.grad.data.add_(noise)

        return norm


# class MemN2N(nn.Module):
#     def __init__(self,
#                  batch_size,
#                  vocab_size,
#                  embedding_dim,
#                  sentence_size,
#                  memory_size,
#                  hops,
#                  num_classes,
#                  dropout=0.5,
#                  fix_embed=True,
#                  name='MemN2N'):
#         super(MemN2N, self).__init__()
#         self.batch_size = batch_size
#         self.embedding_dim = embedding_dim
#         self.sentence_size = sentence_size
#         self.memory_size = memory_size
#         self.hops = hops
#         self.num_classes = num_classes
#         self.fix_embed = fix_embed
#         self.name = name
#
#         self.encoding = torch.from_numpy(position_encoding(1, sentence_size * embedding_dim))
#         self.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#
#         self.fc = nn.Linear(3 * embedding_dim, num_classes)
#         self.dropout = nn.Dropout(dropout)
#         # self.fc = nn.Sequential(
#         #     nn.Linear(2 * self.sentence_rnn_size, linear_hidden_dim),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(dropout),
#         #     nn.Linear(linear_hidden_dim, num_classes)
#         # )
#
#     def init_weights(self, embeddings):
#         if embeddings is not None:
#             self.Embedding = self.Embedding.from_pretrained(
#                 embeddings, freeze=self.fix_embed)
#
#     def set_device(self, device):
#         self.encoding = self.encoding.to(device)
#
#     def forward(self, stories, queries):
#         q_emb0 = self.Embedding(queries)
#         q_emb = q_emb0.view(-1, 1, 3 * self.embedding_dim)
#         u_0 = torch.sum(q_emb * self.encoding, 1, keepdim=True).expand(q_emb.size(0), stories.size(1), q_emb.size(-1))
#         u = [u_0]
#
#         for i in range(self.hops):
#             m_emb0 = self.Embedding(stories)
#             m_emb = m_emb0.view(self.batch_size, -1, self.memory_size, 1, 3 * self.embedding_dim)
#             m = torch.sum(m_emb * self.encoding, -2)
#
#             u_temp = u[-1].unsqueeze(-1).transpose(-2, -1)
#             dotted = torch.sum(m * u_temp, -1)
#             probs = F.softmax(dotted, -1)
#             probs_temp = probs.unsqueeze(-1).transpose(-2, -1)
#
#             c_emb0 = self.Embedding(stories)
#             c_emb = c_emb0.view(self.batch_size, -1, self.memory_size, 1, 3 * self.embedding_dim)
#             c_temp = torch.sum(c_emb * self.encoding, -2)
#             c = c_temp.transpose(-2, -1)
#             o_k = torch.sum(c * probs_temp, -1)
#             u_k = u[-1] + o_k
#             u.append(u_k)
#
#         outputs = self.fc(u_k)
#         return outputs
