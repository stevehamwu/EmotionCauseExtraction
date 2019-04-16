#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/2 21:16
# @Author  : Steve Wu
# @Site    : 
# @File    : build_embedding.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import os
import pickle
import numpy as np
from gensim.models import KeyedVectors

data_root = '/data10T/data/wujipeng/'


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, required=True,
                        help="current embedding name")
    parser.add_argument("--pretrained", type=str, required=True,
                        help="pretrained embedding file")
    parser.add_argument("--vocab", type=str, required=True,
                        help="vocab file")
    return vars(parser.parse_args())


def validate_args(args):
    if not os.path.exists(os.path.join(data_root, 'embedding/', args["pretrained"])) or os.path.exists(
            os.path.join(data_root, 'raw_data', args['vocab'])):
        raise FileNotFoundError


def load_pretrained(pretrained_file):
    print('Loading pretrained embeddings')
    model = KeyedVectors.load_word2vec_format(pretrained_file, datatype=np.float32)
    size = len(model.vocab)
    dim = model.vector_size
    print('{}: size: {} dim: {}'.format(pretrained_file, size, dim))
    return model, dim


def load_vocab(vocab_file):
    print('Loading vocab')
    with open(vocab_file) as f:
        word_unk = f.readline().strip()
        vocab = [word_unk] + f.readline().strip().split(' ')
    print('Vocab size: {}'.format(len(vocab) + 1))
    return vocab


def build_embedding(pretrained, vocab, dim):
    print('Building embedding')
    cnt = 0
    embed_unk = np.random.normal(loc=0., scale=0.1, size=dim)
    embeddings = [np.zeros(dim), embed_unk]
    for word in vocab[1:]:
        if pretrained.vocab.get(word):
            embeddings.append(pretrained.get_vector(word))
            cnt += 1
        else:
            embeddings.append(embed_unk)
    embeddings = np.array(embeddings, dtype=np.float32)
    print('Embedding rate: {:.2f}%'.format(cnt / (len(vocab) + 1) * 100))
    print('Embedding shape: {}'.format(embeddings.shape))
    return embeddings


if __name__ == '__main__':
    args = parse_args()
    pretrained, dim = load_pretrained(os.path.join(data_root, 'embedding', args["pretrained"]))
    vocab = load_vocab(os.path.join(data_root, 'ec/data/raw_data', args["vocab"]))
    embedding = build_embedding(pretrained, vocab, dim)
    pickle.dump(embedding,
                open(os.path.join(data_root, 'ec/data/embedding', '{}_{}d.pkl'.format(args["name"], dim)), 'wb'))
