#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/22 9:38
# @Author  : Steve Wu
# @Site    : 
# @File    : attention.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class BertSingleAttention(nn.Module):
    """
    Compute 'Scaled Dot Product BertSingleAttention'
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = BertSingleAttention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class GeneralAttention(nn.Module):
    """
    Compute 'Scaled General Product BertSingleAttention'
    """

    def __init__(self, hidden_size):
        super(GeneralAttention, self).__init__()
        self.linear = nn.Linear(2 * hidden_size, 2 * hidden_size)

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(self.linear(query), key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class ConcatAttention(nn.Module):
    """
    Compute 'Scaled Concat Product BertSingleAttention'
    """

    def __init__(self, hidden_size):
        super(ConcatAttention, self).__init__()
        self.linear = nn.Linear(2 * 2 * hidden_size, 2 * hidden_size)
        self.proj = nn.Linear(2 * hidden_size, 1)

    def forward(self, query, key, value, mask=None, dropout=None):
        concat = self.linear(torch.cat([query, key], dim=-1)).tanh()
