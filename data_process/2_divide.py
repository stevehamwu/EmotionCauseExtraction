#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/29 10:24
# @Author  : Steve Wu
# @Site    : 
# @File    : 2_divide.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import random
import os


data_root = '/data/wujipeng/ec/data/'


def sampling(total, test_size):
    test_id = set()
    for i in range(1, total + 1):
        if random.random() > (1 - test_size):
            test_id.add(i)
    rate = len(test_id) / total
    if rate > test_size and rate - 0.1 < 0.0003:
        print('Test rate: {}'.format(rate))
        print('Test num: {}'.format(len(test_id)))
        return test_id
    else:
        return sampling(total, test_size)


def divide(path):
    if not os.path.exists(path):
        os.makedirs(path)
    elif os.path.exists(os.path.join(path, 'train_set.txt')) and os.path.exists(os.path.join(path, 'val_set.txt')):
        print('Data already exists')
        return
    test_id = sampling(2105, 0.1)
    with open(os.path.join(data_root, 'han', 'processed_data.csv'), 'r') as f:
        with open(os.path.join(path, 'train_set.txt'), 'w') as ft:
            with open(os.path.join(path, 'val_set.txt'), 'w') as fd:
                f.readline()
                for line in f.readlines():
                    sid = int(line.strip().split(',')[0])
                    fw = fd if sid in test_id else ft
                    fw.write(line)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--session", type=str, required=True,
                        help="current session name, distinguishing models between different hparams.(no suffix)")
    parser.add_argument("--exp", type=str, required=True,
                        help="current session exp, distinguishing models between different hparams.(no suffix)")
    return vars(parser.parse_args())


def validate_args(args):
    if args["session"] is not None:
        if not os.path.exists(os.path.join(data_root, args["session"], args["session"] + '.' + args["exp"])):
            os.makedirs(os.path.join(data_root, args["session"], args["session"] + '.' + args["exp"]))


if __name__ == '__main__':
    args = parse_args()
    if args["exp"] == 'test':
        print('Load test data')
    else:
        validate_args(args)
        divide(os.path.join(data_root, args["session"], args["session"] + '.' + args["exp"]))
