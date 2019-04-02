#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/22 9:50
# @Author  : Steve Wu
# @Site    :
# @File    : han.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import random
import numpy as np


class ECDataLoader:
    def __init__(self, dataset, clause_length, batch_size=16, shuffle=True, sort=True, auto_refresh=True,
                 collate_fn=None):
        self.dataset = dataset
        self.size = len(dataset)
        self.clause_length = clause_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sort = sort
        self.batches = None
        self.num_batches = None
        self.auto_refresh = auto_refresh
        self.collate_fn = collate_fn
        self.inst_count = 0
        self.batch_count = 0
        self._curr_batch = 0
        self._curr_num_insts = None
        if self.sort:
            self.dataset = sorted(dataset, key=lambda item: len(item[1]))
        else:
            self.dataset = list(dataset)
        if self.auto_refresh:
            self.refresh()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.num_batches

    def next(self):
        if self._curr_batch and self._curr_batch + 1 >= self.num_batches:
            if self.auto_refresh:
                self.refresh()
            raise StopIteration
        data = self.get_data()
        return data

    def get_data(self):
        self._curr_batch = (self._curr_batch + 1) if self._curr_batch is not None else 0
        self._curr_num_insts = len(self.batches[self._curr_batch])

        self.inst_count += self._curr_num_insts
        self.batch_count += 1
        data = self.batches[self._curr_batch]
        if self.collate_fn:
            data = self.collate_fn(data, clause_length=self.clause_length, batch_size=self.batch_size)
        return data

    def refresh(self):
        self.batches = []
        batch_start = 0
        for i in range(self.size // self.batch_size):
            self.batches.append(self.dataset[batch_start: batch_start + self.batch_size])
            batch_start += self.batch_size
        if batch_start != self.size:
            self.batches.append(self.dataset[batch_start:])
        if self.shuffle:
            random.shuffle(self.batches)
        self.num_batches = len(self.batches)
        self._curr_batch = None
        self._curr_num_insts = None

    def state_dict(self):
        """
        Warning! side effect: np_randomstate will influence other
            potion of the program.
        """
        state = {
            "batch_size" : self.batch_size,
            "shuffle" : self.shuffle,
            "batches" : self.batches,
            "num_batches" : self.num_batches,
            "auto_refresh" : self.auto_refresh,
            "inst_count" : self.inst_count,
            "batch_count" : self.batch_count,
            "_curr_batch" : self._curr_batch,
            "_curr_num_insts" : self._curr_num_insts,
            "np_randomstate" : np.random.get_state(),
        }
        return state

    def load_state_dict(self, state):
        """
        Warning! side effect: np_randomstate will influence other
            potion of the program.
        """
        self.batch_size = state["batch_size"]
        self.shuffle = state["shuffle"]
        self.batches = state["batches"]
        self.num_batches = state["num_batches"]
        self.auto_refresh = state["auto_refresh"]
        self.inst_count = state["inst_count"]
        self.batch_count = state["batch_count"]
        self._curr_batch = state["_curr_batch"]
        self._curr_num_insts = state["_curr_num_insts"]
        np.random.set_state(state["np_randomstate"])