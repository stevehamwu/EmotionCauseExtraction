#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/2 15:05
# @Author  : Steve Wu
# @Site    : 
# @File    : train_w2v.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import os
import json
from gensim.models.word2vec import Word2Vec


def load_corpus(mode='hanlp'):
    return [line.strip().split(' ') for line in
            open('/data10T/data/wujipeng/ec/data/embedding/{}_corpus.txt'.format(mode)).readlines()]


def train_w2v(sentences, dim):
    params = {
        'size': dim,
        'alpha': 0.025,
        'window': 5,
        'min_count': 1,
        'max_vocab_size': None,
        'sample': 0.001,
        'seed': 1,
        'workers': 2,
        'min_alpha': 0.0001,
        'sg': 1,
        'hs': 0,
        'negative': 5,
        'ns_exponent': 0.75,
        'cbow_mean': 1,
        'iter': 5,
        'null_word': False,
        'sorted_vocab': True
    }

    return Word2Vec(sentences, **params), params,


def save_model(name, model, params):
    json.dump(params, open(
        os.path.join('/data10T/data/wujipeng/embedding/ec/', '{}_{}d.config'.format(name, model.wv.vector_size)), 'w'),
              ensure_ascii=False, indent=4)
    model.wv.save_word2vec_format(
        os.path.join('/data10T/data/wujipeng/embedding/ec/', '{}_{}d.bin'.format(name, model.wv.vector_size)),
        binary=False)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default='w2v',
                        help="embedding name")
    parser.add_argument("--segmentor", type=str, default='hanlp',
                        help="current session name, distinguishing models between different hparams.(no suffix)")
    parser.add_argument("--dim", type=int, required=True,
                        help="current session exp, distinguishing models between different hparams.(no suffix)")
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    corpus = load_corpus(mode=args['segmentor'])
    model, params = train_w2v(corpus, dim=args["dim"])
    save_model(args['name'], model, params)
