#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/28 20:42
# @Author  : Steve Wu
# @Site    : 
# @File    : 1_preprocess.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import pandas as pd
from pyhanlp import *
from collections import Counter
import numpy as np
import pickle


def hanlp_cut(sentence):
    return [term.word for term in HanLP.segment(sentence)]


def build_corpus(raw_corpus):
    #  Hanlp分词
    print('开始分词...')
    corpus = []
    with open('/data/wujipeng/ec/data/raw_data/corpus.txt', 'w') as f:
        for line in raw_corpus:
            line = ' '.join(hanlp_cut(line))
            corpus.append(line)
            f.write(line + '\n')
    print('完成分词')
    return corpus


def build_vocab(corpus):
    # build vocab
    print('Hanlp 分词')
    vocab_cnt = Counter()
    for line in corpus:
        for word in line.replace(' \x01 ', ' ').split(' '):
            vocab_cnt[word] += 1
    vocab = [word for word, freq in vocab_cnt.most_common()]
    with open('/data/wujipeng/ec/data/raw_data/vocab.txt', 'w') as f:
        f.write('<unk>\n')
        f.write(' '.join(vocab))
    print('词典大小: ', len(vocab) + 2)
    print('保存词典')
    return vocab


def load_embedding():
    # Sogou News 300 dim
    # https://github.com/Embedding/Chinese-Word-Vectors
    print('读取预训练Embbeding')
    word_vec = {}
    with open('/data/wujipeng/embedding/sgns.sogou.word', 'r') as f:
        num, dim = filter(int, f.readline().strip().split(' '))
        for line in f.readlines():
            word, vec = line.strip().split(' ', 1)
            word_vec[word] = [float(v) for v in vec.split(' ')]
    return num, dim, word_vec


def build_embedding(vocab):
    embedding_dim = 300
    embedding = [np.zeros(embedding_dim), np.random.normal(loc=0., scale=0.1, size=embedding_dim)]  # pad unk
    size, dim, word_vec = load_embedding()
    print('Sougou embedding: size {}, dim {}'.format(size, dim))
    cnt = 0
    for word in vocab:
        if word_vec.get(word):
            embedding.append(np.array(word_vec[word]))
            cnt += 1
        else:
            embedding.append(np.random.normal(loc=0., scale=0.1, size=embedding_dim))
    embedding = np.array(embedding)
    print('Embedding rate: {:.2f}%'.format(cnt / len(vocab) * 100))
    pickle.dump(embedding, open('/data/wujipeng/ec/data/embedding/sougou_embedding300d.pkl', 'wb'))
    return embedding


def save_data(data):
    data['id'] = list(range(1, len(data) + 1))
    data[['id', 'clause', 'keyword', 'emotion', 'clause_pos', 'label']].to_csv(
        '/data/wujipeng/ec/data/hrcnn/processed_data.csv', index=False)
    print('保存处理后数据')


def preprocess():
    data = pd.read_csv('/data/wujipeng/ec/data/raw_data/process_data_3.csv', index_col=0)
    corpus = [text.replace(' ', '') for text in data['clause'].tolist()]
    keyword = [' '.join(hanlp_cut(e)) for e in data['keyword'].tolist()]
    corpus = build_corpus(corpus)
    vocab = build_vocab(corpus + keyword)
    embedding = build_embedding(vocab)
    data['clause'] = corpus
    data['keyword'] = keyword
    save_data(data)


if __name__ == '__main__':
    preprocess()
