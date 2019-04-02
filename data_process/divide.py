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
    test_id = sampling(2105, 0.1)
    with open('/data/wujipeng/ec/data/hrcnn/processed_data.csv', 'r') as f:
        with open(os.path.join(path, 'train_set.txt'), 'w') as ft:
            with open(os.path.join(path, 'dev_set.txt'), 'w') as fd:
                f.readline()
                for line in f.readlines():
                    sid = int(line.strip().split(',')[0])
                    fw = fd if sid in test_id else ft
                    fw.write(line)


if __name__ == '__main__':
    divide('/data/wujipeng/ec/data/hrcnn/test/')
