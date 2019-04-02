#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/17 11:22
# @Author  : Steve Wu
# @Site    : 
# @File    : process.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import os


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data.append(line.strip().split(','))
    return data


def pad_sequence(sequences, sequence_length=40, pad=0):
    paded_sequences = []
    for sequence in sequences:
        sl = sequence_length - len(sequence)
        paded_sequences.append(sequence + [pad] * sl)
    return paded_sequences


def pad_memory(sequences, poses, memory_size, sequence_size, pad=0):
    paded_sequences = []
    for sequence, pos in zip(sequences, poses):
        if len(sequence) < sequence_size:
            sequence += [pad] * (sequence_size - len(sequence))
        paded_sequence = [sequence[:sequence_size]]
        for i in range(memory_size-1):
            if i < len(sequence) - sequence_size + 1:
                paded_sequence += [sequence[i:i + 3]]
            elif i == len(sequence) - sequence_size + 1:
                paded_sequence += [[pos] * 3]
            else:
                paded_sequence += [[pad] * 3]
        paded_sequences.append(paded_sequence)
    return paded_sequences
