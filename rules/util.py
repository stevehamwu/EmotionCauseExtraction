#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/13 20:49
# @Author  : Steve Wu
# @Site    : 
# @File    : util.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
# find word


def find_word(keyword, sentence, start=0, end=-1, strict=False):
    """
    word: str 'abc'
    sentence: list ['a', 'b', 'cd']
    return: start_index, end_index: 0, 2
    """
    if not sentence:
        return -1, -1
    if end == -1 or end > len(sentence):
        end = len(sentence)
    if keyword in sentence[start:end]:
        return sentence.index(keyword, start, end), sentence.index(keyword, start, end)
    elif strict:
            return -1, -1
    else:
        s, e = -1, -1
        sentence = sentence[start: end]
        idx = ''.join(sentence).find(keyword)
        if idx >= 0:
            l = -1
            for i, word in enumerate(sentence):
                word = sentence[i]
                l += len(word)
                if l >= idx and s < 0:
                    s = i + start
                if l >= idx+len(keyword)-1:
                    e = i + start
                    break
    return s, e


# rfind word
def rfind_word(keyword, sentence, start=0, end=-1, strict=False):
    """
    word: str 'word'
    sentence: list ['a', 'b', 'cd']
    """
    if not sentence:
        return -1, -1
    if end == -1 or end > len(sentence):
        end = len(sentence)
    s, e = find_word(keyword[::-1], [word[::-1] for word in sentence[::-1]], len(sentence)-end, len(sentence)-start, strict)
    if s == -1 or e == -1:
        return s, e
    return len(sentence)-e-1, len(sentence)-s-1