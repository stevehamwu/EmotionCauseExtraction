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

data_root = '/data/wujipeng/ec/data/'


# def hanlp_cut(sentence):
#     return [term.word for term in HanLP.segment(sentence)]

# 加词性标注
def hanlp_cut(sentence):
    words, natures = [], []
    for term in HanLP.segment(sentence):
        word, nature = term.word, term.nature.toString()
        if '\x01' in word:
            words.append('\x01')
            natures.append('\x01')
        else:
            words.append(word)
            natures.append(nature[0])
    return words, natures


# def build_corpus(raw_corpus):
#     print('开始分词...')    #  Hanlp分词
#
#     corpus = []
#     with open(os.path.join(data_root, 'raw_data', 'corpus.txt'), 'w') as f:
#         for line in raw_corpus:
#             line = ' '.join(hanlp_cut(line))
#             corpus.append(line)
#             f.write(line + '\n')
#     print('完成分词')
#     return corpus


# 加词性标注
def build_corpus(raw_corpus):
    print('开始分词...')  # Hanlp分词
    corpus = []
    natures = []
    with open(os.path.join(data_root, 'raw_data', 'corpus.txt'), 'w') as fc:
        with open(os.path.join(data_root, 'raw_data', 'natures.txt'), 'w') as fn:
            for line in raw_corpus:
                words, tags = hanlp_cut(line)
                words = ' '.join(words)
                tags = ' '.join(tags)
                corpus.append(words)
                natures.append(tags)
                fc.write(words + '\n')
                fn.write(tags + '\n')
    print('完成分词')
    return corpus, natures


def build_vocab(corpus):
    # build vocab
    print('Hanlp 分词')
    vocab_cnt = Counter()
    for line in corpus:
        for word in line.replace(' \x01 ', ' ').split(' '):
            vocab_cnt[word] += 1
    vocab = [word for word, freq in vocab_cnt.most_common()]
    with open(os.path.join(data_root, 'raw_data', 'vocab.txt'), 'w') as f:
        f.write('<unk>\n')
        f.write(' '.join(vocab))
    print('词典大小: ', len(vocab) + 2)
    print('保存词典')
    return vocab


def load_embedding(source="sogou"):
    # Sogou News 300 dim
    # https://github.com/Embedding/Chinese-Word-Vectors
    if source == "sogou":
        pretrained_file = '/data/wujipeng/embedding/sogou/sgns.sogou.word'
    elif source == "ailab":
        pretrained_file = '/data/wujipeng/embedding/Tencent_AILab/Tencent_AILab_ChineseEmbedding.txt'
    print('读取预训练Embbeding')
    word_vec = {}
    with open(pretrained_file, 'r') as f:
        num, dim = filter(int, f.readline().strip().split(' '))
        for line in f.readlines():
            word, vec = line.strip().split(' ', 1)
            word_vec[word] = [float(v) for v in vec.split(' ')]
    return num, dim, word_vec


def build_embedding(vocab):
    source = 'sogou'
    size, embedding_dim, word_vec = load_embedding(source)
    embedding = [np.zeros(embedding_dim), np.random.normal(loc=0., scale=0.1, size=embedding_dim)]  # pad unk
    print('{} embedding: size {}, dim {}'.format(source, size, embedding_dim))
    cnt = 0
    for word in vocab:
        if word_vec.get(word):
            embedding.append(np.array(word_vec[word]))
            cnt += 1
        else:
            embedding.append(np.random.normal(loc=0., scale=0.1, size=embedding_dim))
    embedding = np.array(embedding)
    print('Embedding rate: {:.2f}%'.format(cnt / len(vocab) * 100))
    pickle.dump(embedding,
                open(os.path.join(data_root, 'embedding', '{}_embedding{}d.pkl'.format(source, embedding_dim)), 'wb'))
    return embedding


def build_pos_embedding(poses):
    min_pos = min([min(pos) for pos in poses])
    max_pos = max([max(pos) for pos in poses])
    poses = [' '.join([str(p - min_pos + 1) for p in pos]) for pos in poses]
    pos_size = max_pos - min_pos + 1
    print('pos_size:{}'.format(pos_size + 1))
    pos_embedding_dim = 300
    pos_embedding = np.concatenate((np.zeros((1, pos_embedding_dim)), np.random.randn(pos_size, pos_embedding_dim)))
    pickle.dump(pos_embedding, open(os.path.join(data_root, 'embedding', 'pos_embedding.pkl'), 'wb'))
    return poses


def save_data(data):
    if not os.path.exists(os.path.join(data_root, 'han')):
        os.makedirs(os.path.join(data_root, 'han'))
    data['id'] = list(range(1, len(data) + 1))
    data[['id', 'clause', 'nature', 'keyword', 'emotion', 'clause_pos', 'label']].to_csv(
        os.path.join(data_root, 'han', 'processed_data.csv'), index=False)
    print('保存处理后数据')


def preprocess():
    data = pd.read_csv(os.path.join(data_root, 'raw_data', 'process_data_3.csv'), index_col=0)
    corpus = [text.replace(' ', '') for text in data['clause'].tolist()]
    keyword = [' '.join(hanlp_cut(e)[0]) for e in data['keyword'].tolist()]
    poses = [list(map(int, pos.split(' '))) for pos in data['clause_pos'].tolist()]

    # corpus = build_corpus(corpus)
    corpus, natures = build_corpus(corpus)
    vocab = build_vocab(corpus + keyword)
    build_embedding(vocab)
    poses = build_pos_embedding(poses)

    data['clause'] = corpus
    data['nature'] = natures
    data['keyword'] = keyword
    data['clause_pos'] = poses
    save_data(data)


if __name__ == '__main__':
    preprocess()
