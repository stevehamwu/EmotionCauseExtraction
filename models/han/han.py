#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/22 9:39
# @Author  : Steve Wu
# @Site    : 
# @File    : han.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import pickle
import numpy as np
import torch
import torch.nn as nn
from .attention import *
from .word_model import *
from .sentence_model import *


class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 embedding_dim,
                 word_model,
                 sentence_model,
                 dropout=0.5,
                 fix_embed=True,
                 name='HAN'):
        super(HierarchicalAttentionNetwork, self).__init__()
        self.sentence_rnn_size = sentence_model['args']['rnn_size']
        self.num_classes = num_classes
        self.fix_embed = fix_embed
        self.name = name

        self.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_rnn = eval(word_model['class'])(**word_model['args'])
        self.sentence_rnn = eval(sentence_model['class'])(**sentence_model['args'])
        self.fc = nn.Linear(2 * self.sentence_rnn_size, num_classes)
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
        documents, word_attn = self.word_rnn(inputs, queries)
        outputs, sentence_attn = self.sentence_rnn(documents, poses)
        # outputs = self.fc(outputs)
        outputs = self.fc(self.dropout(outputs))
        return outputs, word_attn, sentence_attn


class HierarchicalAttentionNetworkV2(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 embedding_dim,
                 word_model,
                 sentence_model,
                 dropout=0.5,
                 fix_embed=True,
                 name='HAN'):
        super(HierarchicalAttentionNetworkV2, self).__init__()
        self.word_rnn_size = word_model['args']['rnn_size']
        self.sentence_rnn_size = sentence_model['args']['rnn_size']
        self.num_classes = num_classes
        self.fix_embed = fix_embed
        self.name = name

        self.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_rnn = eval(word_model['class'])(**word_model['args'])
        self.fc1 = nn.Linear(2 * self.word_rnn_size, num_classes)
        self.dropout1 = nn.Dropout(dropout)
        self.sentence_rnn = eval(sentence_model['class'])(**sentence_model['args'])
        self.fc2 = nn.Linear(2 * self.sentence_rnn_size, num_classes)
        self.dropout2 = nn.Dropout(dropout)
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
        documents, word_attn = self.word_rnn(inputs, queries)
        word_outputs = self.fc1(self.dropout1(documents))
        outputs, sentence_attn = self.sentence_rnn(documents, poses)
        # outputs = self.fc(outputs)
        outputs = self.fc2(self.dropout2(outputs))
        return word_outputs, outputs, word_attn, sentence_attn


class HierarchicalAttentionNetworkV3(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 embedding_dim,
                 word_model,
                 sentence_model,
                 dropout=0.5,
                 fix_embed=True,
                 name='HAN'):
        super(HierarchicalAttentionNetworkV3, self).__init__()
        self.word_rnn_size = word_model['args']['rnn_size']
        self.sentence_rnn_size = sentence_model['args']['rnn_size']
        self.num_classes = num_classes
        self.fix_embed = fix_embed
        self.name = name

        self.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_rnn = eval(word_model['class'])(**word_model['args'])
        self.sentence_rnn = eval(sentence_model['class'])(**sentence_model['args'])
        self.fc = nn.Linear(2 * self.word_rnn_size + 2 * self.sentence_rnn_size, num_classes)
        self.dropout = nn.Dropout(dropout)
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
        outputs = self.fc(self.dropout(s_c))
        return outputs, word_attn, sentence_attn


class HierarchicalAttentionNetworkV3ELMo(nn.Module):
    def __init__(self,
                 num_classes,
                 elmo_dim,
                 sentence_model,
                 dropout=0.5,
                 fix_embed=True,
                 name='HAN'):
        super(HierarchicalAttentionNetworkV3ELMo, self).__init__()
        self.sentence_rnn_size = sentence_model['args']['rnn_size']
        self.num_classes = num_classes
        self.fix_embed = fix_embed
        self.name = name

        self.sentence_rnn = eval(sentence_model['class'])(**sentence_model['args'])
        self.fc = nn.Linear(elmo_dim + 2 * self.sentence_rnn_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Sequential(
        #     nn.Linear(2 * self.sentence_rnn_size, linear_hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(linear_hidden_dim, num_classes)
        # )

    def forward(self, sentences, keywords, poses):
        outputs, sentence_attn = self.sentence_rnn(sentences, poses)
        # outputs = self.fc(outputs)
        s_c = torch.cat((sentences, outputs), dim=-1)
        outputs = self.fc(self.dropout(s_c))
        return outputs, None, sentence_attn


class HierarchicalAttentionNetworkV3Init(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 embedding_dim,
                 word_model,
                 sentence_model,
                 dropout=0.5,
                 fix_embed=True,
                 name='HAN'):
        super(HierarchicalAttentionNetworkV3Init, self).__init__()
        self.word_rnn_size = word_model['args']['rnn_size']
        self.sentence_rnn_size = sentence_model['args']['rnn_size']
        self.num_classes = num_classes
        self.fix_embed = fix_embed
        self.name = name

        self.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_rnn = eval(word_model['class'])(**word_model['args'])
        self.sentence_rnn = eval(sentence_model['class'])(**sentence_model['args'])
        self.fc = nn.Linear(2 * self.word_rnn_size + 2 * self.sentence_rnn_size, num_classes)
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
        for n, para in self.named_parameters():
            if 'reverse' in n:
                continue
            elif 'bias' in n:
                nn.init.zeros_(para)
            elif 'rnn' in n:
                nn.init.orthogonal_(para)
            elif 'fc' in n:
                nn.init.xavier_uniform_(para)

    def forward(self, sentences, keywords, poses):
        inputs = self.Embedding(sentences)
        queries = self.Embedding(keywords)
        documents, word_attn = self.word_rnn(inputs, queries)
        outputs, sentence_attn = self.sentence_rnn(documents, poses)
        # outputs = self.fc(outputs)
        s_c = torch.cat((documents, outputs), dim=-1)
        outputs = self.fc(self.dropout(s_c))
        return outputs, word_attn, sentence_attn


class HierarchicalAttentionNetworkV3InitV2(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 embedding_dim,
                 word_model,
                 sentence_model,
                 dropout=0.5,
                 fix_embed=True,
                 name='HAN'):
        super(HierarchicalAttentionNetworkV3InitV2, self).__init__()
        self.word_rnn_size = word_model['args']['rnn_size']
        self.sentence_rnn_size = sentence_model['args']['rnn_size']
        self.num_classes = num_classes
        self.fix_embed = fix_embed
        self.name = name

        self.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_rnn = eval(word_model['class'])(**word_model['args'])
        self.sentence_rnn = eval(sentence_model['class'])(**sentence_model['args'])
        self.fc = nn.Linear(2 * self.word_rnn_size + 2 * self.sentence_rnn_size, num_classes)
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
        for n, para in self.named_parameters():
            if 'reverse' in n:
                continue
            elif 'bias' in n:
                nn.init.zeros_(para)
            elif 'rnn' in n:
                nn.init.orthogonal_(para)

    def forward(self, sentences, keywords, poses):
        inputs = self.Embedding(sentences)
        queries = self.Embedding(keywords)
        documents, word_attn = self.word_rnn(inputs, queries)
        outputs, sentence_attn = self.sentence_rnn(documents, poses)
        # outputs = self.fc(outputs)
        s_c = torch.cat((documents, outputs), dim=-1)
        outputs = self.fc(self.dropout(s_c))
        return outputs, word_attn, sentence_attn


class HierarchicalAttentionNetworkV4(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 embedding_dim,
                 hidden_size,
                 word_model,
                 sentence_model,
                 dropout=0.5,
                 fix_embed=True,
                 name='HAN'):
        super(HierarchicalAttentionNetworkV4, self).__init__()
        self.word_rnn_size = word_model['args']['rnn_size']
        self.sentence_rnn_size = sentence_model['args']['rnn_size']
        self.num_classes = num_classes
        self.fix_embed = fix_embed
        self.name = name

        self.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.linear = nn.Linear(embedding_dim, hidden_size)
        self.word_rnn = eval(word_model['class'])(**word_model['args'])
        self.sentence_rnn = eval(sentence_model['class'])(**sentence_model['args'])
        self.fc = nn.Linear(2 * self.word_rnn_size + 2 * self.sentence_rnn_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Sequential(
        #     nn.Linear(2 * self.sentence_rnn_size, linear_hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(linear_hidden_dim, num_classes)
        # )

    def init_weights(self, embeddings):
        if embeddings is not None:
            self.Embedding = self.Embedding.from_pretrained(embeddings)

    def forward(self, sentences, keywords, poses):
        inputs = self.linear(self.Embedding(sentences))
        queries = self.linear(self.Embedding(keywords))
        documents, word_attn = self.word_rnn(inputs, queries)
        outputs, sentence_attn = self.sentence_rnn(documents, poses)
        # outputs = self.fc(outputs)
        s_c = torch.cat((documents, outputs), dim=-1)
        outputs = self.fc(self.dropout(s_c))
        return outputs, word_attn, sentence_attn


# Author: Robert Guthrie
# https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class HierarchicalAttentionNetworkCRF(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 tagset_size,
                 embedding_dim,
                 word_model,
                 sentence_model,
                 dropout=0.5,
                 fix_embed=True,
                 name='HAN'):
        super(HierarchicalAttentionNetworkCRF, self).__init__()
        self.word_rnn_size = word_model['args']['rnn_size']
        self.sentence_rnn_size = sentence_model['args']['rnn_size']
        self.num_classes = num_classes
        self.tagset_size = tagset_size
        self.fix_embed = fix_embed
        self.name = name

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_rnn = eval(word_model['class'])(**word_model['args'])
        self.sentence_rnn = eval(sentence_model['class'])(**sentence_model['args'])
        self.fc = nn.Linear(2 * self.word_rnn_size + 2 * self.sentence_rnn_size, tagset_size)
        self.dropout = nn.Dropout(dropout)

    def init_weights(self, embeddings):
        if embeddings is not None:
            self.Embedding = self.Embedding.from_pretrained(embeddings)

    def set_device(self, device):
        self.device = device

    def _forward_alg(self, feats, masks):
        alpha = torch.zeros(1).to(self.device)
        forward_vars = torch.FloatTensor().to(self.device)
        # Iterate through the sentence
        for feat, mask in zip(feats, masks):
            # Do the forward algorithm to compute the partition function
            init_alphas = torch.full((1, self.tagset_size), -10000.).to(self.device)
            # START_TAG has all of the score.
            init_alphas[0][2] = 0.

            # Wrap in a variable so that we will get automatic backprop
            forward_var = init_alphas

            feat = torch.masked_select(feat, mask.unsqueeze(-1).expand_as(feat)).view(-1, feat.size(-1))
            for f in feat:
                alphas_t = []  # The forward tensors at this timestep
                for next_tag in range(self.tagset_size):
                    # broadcast the emission score: it is the same regardless of
                    # the previous tag
                    emit_score = f[next_tag].view(
                        1, -1).expand(1, self.tagset_size)
                    # the ith entry of trans_score is the score of transitioning to
                    # next_tag from i
                    trans_score = self.transitions[next_tag].view(1, -1)
                    # The ith entry of next_tag_var is the value for the
                    # edge (i -> next_tag) before we do log-sum-exp
                    next_tag_var = forward_var + trans_score + emit_score
                    # The forward variable for this tag is log-sum-exp of all the
                    # scores.
                    alphas_t.append(log_sum_exp(next_tag_var).view(1))
                forward_var = torch.cat(alphas_t).view(1, -1)
                forward_vars = torch.cat((forward_vars, forward_var))
            terminal_var = forward_var + self.transitions[-1]
            alpha += log_sum_exp(terminal_var)
        return alpha, forward_vars[:, :2]

    def _get_lstm_features(self, sentences, keywords, poses):

        inputs = self.Embedding(sentences)
        queries = self.Embedding(keywords)
        documents, word_attn = self.word_rnn(inputs, queries)
        outputs, sentence_attn = self.sentence_rnn(documents, poses)
        # outputs = self.fc(outputs)
        s_c = torch.cat((documents, outputs), dim=-1)
        lstm_feats = self.fc(self.dropout(s_c))
        return lstm_feats, word_attn, sentence_attn

    def _score_sentence(self, feats, tags, masks):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(self.device)
        for feat, tag, mask in zip(feats, tags, masks):
            feat = torch.masked_select(feat, mask.unsqueeze(-1).expand_as(feat)).view(-1, feat.size(-1))
            tag = torch.masked_select(tag, mask)
            tag = torch.cat((torch.LongTensor([2]).to(self.device), tag))
            for i, f in enumerate(feat):
                score += self.transitions[tag[i + 1], tag[i]] + f[tag[i + 1]]
            score += self.transitions[-1, tag[-1]]
        return score

    def _viterbi_decode(self, feats):
        path_score = torch.zeros(1).to(self.device)
        best_paths = []
        forward_vars = torch.FloatTensor().to(self.device)

        for feat in feats:
            backpointers = []

            # Initialize the viterbi variables in log space
            init_vvars = torch.full((1, self.tagset_size), -10000.).to(self.device)
            init_vvars[0][2] = 0

            # forward_var at step i holds the viterbi variables for step i-1
            forward_var = init_vvars
            for f in feat:
                bptrs_t = []  # holds the backpointers for this step
                viterbivars_t = []  # holds the viterbi variables for this step

                for next_tag in range(self.tagset_size):
                    # next_tag_var[i] holds the viterbi variable for tag i at the
                    # previous step, plus the score of transitioning
                    # from tag i to next_tag.
                    # We don't include the emission scores here because the max
                    # does not depend on them (we add them in below)
                    next_tag_var = forward_var + self.transitions[next_tag]
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                # Now add in the emission scores, and assign forward_var to the set
                # of viterbi variables we just computed
                forward_var = (torch.cat(viterbivars_t) + f).view(1, -1)
                forward_vars = torch.cat((forward_vars, forward_var))
                backpointers.append(bptrs_t)

            # Transition to STOP_TAG
            terminal_var = forward_var + self.transitions[-1]
            best_tag_id = argmax(terminal_var)
            path_score += terminal_var[0][best_tag_id]

            # Follow the back pointers to decode the best path.
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            # Pop off the start tag (we dont want to return that to the caller)
            start = best_path.pop()
            assert start == 2  # Sanity check
            best_path.reverse()
            best_paths.append(best_path)
        forward_vars = forward_vars.view_as(feats)[:, :, :2]
        return path_score, torch.LongTensor(best_paths).to(self.device), forward_vars

    def neg_log_likelihood(self, sentences, keywords, poses, tags, masks):
        feats, word_attn, sentence_attn = self._get_lstm_features(sentences, keywords, poses)
        forward_score, forward_probs = self._forward_alg(feats, masks)
        gold_score = self._score_sentence(feats, tags, masks)
        return forward_score - gold_score, forward_probs, word_attn, sentence_attn

    def forward(self, sentences, keywords, poses):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats, word_attn, sentence_attn = self._get_lstm_features(sentences, keywords, poses)

        # Find the best path, given the features.
        score, tag_seq, tag_probs = self._viterbi_decode(lstm_feats)
        return score, tag_seq, tag_probs, word_attn, sentence_attn


class HierarchicalAttentionNetworkRule(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 embedding_dim,
                 word_model,
                 sentence_model,
                 dropout=0.5,
                 fix_embed=True,
                 name='HAN'):
        super(HierarchicalAttentionNetworkRule, self).__init__()
        self.word_rnn_size = word_model['args']['rnn_size']
        self.sentence_rnn_size = sentence_model['args']['rnn_size']
        self.num_classes = num_classes
        self.fix_embed = fix_embed
        self.name = name

        self.Embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_rnn = eval(word_model['class'])(**word_model['args'])
        self.sentence_rnn = eval(sentence_model['class'])(**sentence_model['args'])
        self.fc = nn.Linear(2 * self.word_rnn_size + 2 * self.sentence_rnn_size, num_classes)
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
        documents, word_attn = self.word_rnn(inputs, queries)
        outputs, sentence_attn = self.sentence_rnn(documents, poses)
        # outputs = self.fc(outputs)
        s_c = torch.cat((documents, outputs), dim=-1)
        outputs = self.fc(self.dropout(s_c))
        return outputs, word_attn, sentence_attn
