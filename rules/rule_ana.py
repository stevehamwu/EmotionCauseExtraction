#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/8 11:22
# @Author  : Steve Wu
# @Site    : 
# @File    : rule_ana.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
from .util import *
import numpy as np
import pandas as pd


def test_rules(data=pd.read_csv('/data/wujipeng/ec/data/han/processed_data.csv'), rule_range=range(1, 16), strict=True):
    pos0 = 69
    rules = np.zeros((len(data), 15), dtype=int)
    rule_preds = []
    masks = []
    if min(rule_range) in [14, 15]:
        rule_list = list(range(1, 14)) + list(rule_range)
    else:
        rule_list = rule_range
    for i, row in data.iterrows():
        clauses = [clause.strip().split(' ') for clause in row['clause'].split('\x01')]
        natures = [nature.strip().split(' ') for nature in row['nature'].split('\x01')]
        keyword = row['keyword'].replace(' ', '')
        emotion = row['emotion']
        clause_pos = [int(pos) - pos0 for pos in row['clause_pos'].split(' ')]
        rule_pred = np.zeros(len(clauses), dtype=int)
        mask = np.array([np.zeros_like(np.array(clause), dtype=int) for clause in clauses])

        f = clause_pos.index(0)
        K = keyword

        for r in rule_list:
            if r in [6, 7, 8, 12, 13]:
                continue
            # constraints
            if r == 4:
                # 0,1: Happiness 2:Anger 3: Sadness 4: Fear 5:Disgust 6:Surprise
                if emotion not in [0, 1, 4, 6]:
                    continue
            if r in [14, 15]:
                if sum(rules[i]) > 0:
                    if min(rule_range) in [14, 15]:
                        rules[i] = np.zeros(15)
                        rule_pred = np.zeros(len(clauses), dtype=int)
                        mask = np.array([np.zeros_like(np.array(clause), dtype=int) for clause in clauses])
                    continue
            found = eval('rule{}'.format(r))(f, clauses, natures, K, strict)
            if found:
                rules[i][r - 1] = 1
                if found[0] == 'F':
                    rule_pred[f] = 1
                elif found[0] == 'B':
                    rule_pred[f - 1] = 1
                elif found[0] == 'A':
                    rule_pred[f + 1] = 1
                mask = found[-1]
                break
        masks.append(mask)
        rule_preds.append(rule_pred)
    return rules, rule_preds, masks


markers = [
    ['让', '令', '使'],
    ['想到', '想起', '一想', '想来', '说到', '说起', '一说', '讲起', '谈到', '谈起', '提到', '提起', '一提'],
    ['说', '的说', '道'],
    ['看', '看到', '看见', '见到', '见', '眼看', '听', '听到', '听说', '知道', '得知', '获知', '获悉', '发现', '发觉', '有'],
    ['为', '为了', '对', '对于'],
    ['因', '因为', '由于'],
    ['的是', '于', '能']
]


def rule1(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)
    masks = np.array([np.zeros_like(np.array(clause), dtype=int) for clause in clauses])

    # constraint
    constraint_words = ['的', '的是', '是']
    for word in constraint_words:
        c_s, c_e = find_word(word, F, k_e, -1)
        if c_s > -1:
            return
        c_s, c_e = find_word(word, A)
        if c_s > -1:
            return

    cues = markers[0]  # I

    for word in cues:
        # 找线索词
        i_s, i_e = find_word(word, F, 0, k_s + 1, strict=strict)
        if B is not None:
            masks[b][:] = 1
        masks[f][:i_s] = 1
        if i_s > -1:
            I = word
            # 找情感对象
            e_s, e_e = find_word('n', FN, i_e, k_s + 1)
            if e_s > -1:
                E = F[e_s: e_e + 1]
                # 在当前句找情感原因
                c_s, c_e = rfind_word('v', FN, 0, i_s + 1)
                if c_s > -1 and i_s > 1:
                    C = F[c_s: c_e + 1]
                    return 'F', I, E, C, masks
                else:
                    # 在前一句找情感原因
                    c_s, c_e = rfind_word('v', BN)
                    if c_s > -1 and len(B) - c_e + i_s > 1:
                        C = B[c_s: c_e + 1]
                        return 'B', I, E, C, masks
    return


def rule2(f, clauses, natures, K, strict=True):
    b, a = f-1, f+1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)
    masks = np.array([np.zeros_like(np.array(clause), dtype=int) for clause in clauses])

    cues = markers[1] + markers[3] + markers[4] + markers[5] # I
    # constraint
    cues.remove('有')
    for word in cues:
        # 在当前句找线索词
        i_s, i_e = find_word(word, F, 0, k_s+1, strict=strict)
        if i_s > -1:
            masks[f][i_e+1: k_s] = 1
            I = F[i_s: i_e+1]
            # 找情感原因
            c_s, c_e = find_word('v', FN, i_e+1, k_s)
            if c_s > -1 and find_word('n', FN, i_e, k_s+1)[0] > -1:
                C = F[c_s: c_e+1]
                # 在当前句找情感对象
                e_s, e_e = rfind_word('n', FN, 0, i_s+1)
                if e_s > -1:
                    E = F[e_s: e_e+1]
                else:
                    # 在前一句找情感对象
                    e_s, e_e = rfind_word('n', BN)
                    if e_s > -1:
                        E = B[e_s: e_e+1]
                if e_s > -1:
                    return 'F', I, E, C, masks
        else:
            #在前一句找线索词
            i_s, i_e = find_word(word, B, strict=True)
            masks[f][:k_s] = 1
            if i_s > -1:
                masks[b][i_e+1:] = 1
                I = B[i_s: i_e+1]
                # 找情感对象
                e_s, e_e = rfind_word('n', BN, 0, i_s+1)
                if e_s > -1:
                    E = B[e_s: e_e+1]
                    # 在前一句找情感原因
                    c_s, c_e = find_word('v', BN, i_e)
                    if c_s > -1 and find_word('v', BN, i_e)[0] > -1:
                        C = B[c_s: c_e+1]
                        return 'B', I, E, C, masks
                    else:
                        # 在当前句找情感原因
                        c_s, c_e = find_word('v', FN, 0, k_s)
                        if c_s > -1 and find_word('v', FN, 0, k_s)[0] > -1:
                            C = F[c_s: c_e+1]
                            return 'F', I, E, C, masks
    return


def rule3(f, clauses, natures, K, strict=True):
    b, a = f-1, f+1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)
    masks = np.array([np.zeros_like(np.array(clause), dtype=int) for clause in clauses])

    cues = markers[1] + markers[3] + markers[4] + markers[5] # I
    for word in cues:
        # 找线索词
        i_s, i_e = find_word(word, B, strict=strict)
        if i_s > -1:
            masks[b][i_e+1:] = 1
            I = B[i_s: i_e+1]
            # 找情感原因
            c_s, c_e = find_word('v', BN, i_e if BN[i_e] != 'v' else i_e+1)
            if c_s > -1 and len(B)-i_e>3:
                C = B[c_s: c_e+1]
                # 找情感对象
                e_s, e_e = find_word('n', FN, 0, k_s+1)
                E = F[e_s: e_e+1]
                if e_s > -1:
                    return 'B', I, E, C, masks
    return


def rule4(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)
    masks = np.array([np.zeros_like(np.array(clause), dtype=int) for clause in clauses])

    cues = markers[3] + markers[6]

    for word in cues:
        # 找线索词
        i_s, i_e = find_word(word, F, k_e + 1, strict=strict)
        masks[f][i_e+1:] += 1
        if A is not None:
            masks[a][:] = 1
        if i_s > -1:
            I = F[i_s: i_e + 1]
            # 在当前句找情感对象
            e_s, e_e = rfind_word('n', FN, 0, k_s + 1)
            if e_s > -1:
                E = F[e_s: e_e + 1]
            else:
                # 在前一句找情感对象
                e_s, e_e = find_word('n', BN)
                if e_s > -1:
                    E = B[e_s: e_e + 1]
            if e_s > -1:
                # 在当前句找情感原因
                c_s, c_e = find_word('v', FN, i_e)
                if c_s > -1 and len(F) - i_e > 1:
                    C = F[c_s: c_e + 1]
                    return 'F', I, E, C, masks
                else:
                    # 在下一句找情感原因
                    c_s, c_e = find_word('v', AN)
                    if c_s > -1 and len(F) - i_e + c_s > 1:
                        C = A[c_s: c_e + 1]
                        return 'A', I, E, C, masks
    return


def rule5(f, clauses, natures, K, strict=True):

    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)
    masks = np.array([np.zeros_like(np.array(clause), dtype=int) for clause in clauses])

    cues = markers[5]

    for word in cues:
        # 找线索词
        i_s, i_e = find_word(word, A, strict=strict)
        if i_s > -1:
            masks[a][i_e+1:] = 1
            I = A[i_s: i_e + 1]
            # 找情感对象
            e_s, e_e = find_word('n', FN, 0, k_s + 1)
            if e_s > -1:
                E = F[e_s: e_e + 1]
                # 找情感原因
                c_s, c_e = find_word('nvn', AN, i_e)
                if c_s > -1:
                    C = A[c_s: c_e + 1]
                    return 'A', I, E, C, masks
    return


# params f, clauses, natures, K,
def rule6(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)
    masks = np.array([np.zeros_like(np.array(clause), dtype=int) for clause in clauses])

    cues = markers[0]

    for word in cues:
        # 找线索词
        i_s, i_e = find_word(word, F, strict=strict)
        if i_s > -1:
            I = F[i_s: i_e + 1]
            # 找情感对象
            e_s, e_e = find_word('n', FN, i_e, k_s + 1)
            masks[f][k_e + 1:] = 1
            if e_s > -1:
                E = F[e_s: e_e + 1]
                # 在当前句找情感原因
                c_s, c_e = find_word('v', FN, k_e)
                if c_s > -1 and len(F) - k_e > 1:
                    C = F[c_s: c_e + 1]
                    return 'F', I, E, C, masks
                else:
                    # 在下一句找情感原因
                    c_s, c_e = find_word('v', AN)
                    if c_s > -1 and len(F) - k_e + c_s > 1:
                        masks[a][:] = 1
                        C = A[c_s: c_e + 1]
                        return 'A', I, E, C, masks
    return


# params f, clauses, natures, K,
def rule7(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)
    masks = np.array([np.zeros_like(np.array(clause), dtype=int) for clause in clauses])

    # 找到第二个越
    i_s, i_e = rfind_word('越', F, 0, k_s + 1)
    if i_s >= k_s - 1:
        # 找情感原因
        c_s, c_e = rfind_word('v', FN, 0, i_s + 1)
        if c_s > -1:
            C = F[c_s: c_e + 1]
            i_s, i_e = rfind_word('越', F, 0, c_s + 1)
            if i_s >= c_s - 1:
                I = '越..越'
                # 在当前句找情感对象
                e_s, e_e = rfind_word('n', FN, 0, i_s + 1)
                if e_s > -1:
                    E = F[e_s: e_e + 1]
                else:
                    # 在前一句找情感对象
                    e_s, e_e = rfind_word('n', BN)
                    if e_s > -1:
                        E = B[e_s: e_e + 1]
                if e_s > -1:
                    return 'F', I, E, C, masks
    return


def rule8(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)
    masks = np.array([np.zeros_like(np.array(clause), dtype=int) for clause in clauses])
    masks[f][k_e + 1:] = 1

    # 找情感对象
    e_s, e_e = find_word('n', FN, 0, k_s + 1)
    if e_s > -1:
        E = F[e_s: e_e + 1]
        # 找情感原因
        c_s, c_e = find_word('nvn', FN, k_e + 1)
        if c_s > -1:
            C = F[c_s: c_e + 1]
            return 'F', '', E, C, masks
    return


def rule9(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)
    masks = np.array([np.zeros_like(np.array(clause), dtype=int) for clause in clauses])

    cues = markers[3]
    for word in cues:
        # 找线索词
        i_s, i_e = rfind_word(word, F, 0, k_s + 1, strict=strict)
        masks[f][i_e+1: k_s] = 1
        if i_s > -1 and k_s - i_e > 1:
            I = F[i_s: i_e + 1]
            c_s, c_e = find_word('v', FN, i_e + 1, k_s + 1)
            if c_s > -1 and find_word('n', FN, i_e + 1, k_s + 1)[0] > -1:
                C = F[c_s: c_e + 1]
                # 找情感对象
                e_s, e_e = rfind_word('n', FN, 0, i_s + 1)
                if e_s > -1:
                    E = F[e_s: e_e + 1]
                    return 'F', I, E, C, masks
    return


def rule10(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)
    masks = np.array([np.zeros_like(np.array(clause), dtype=int) for clause in clauses])

    # 找情感对象
    e_s, e_e = find_word('n', FN, k_e)
    if e_s > -1:
        E = F[e_s: e_e + 1]
        # 找情感原因开端
        c_s, c_e = find_word('nvn', FN, e_e)
        if c_s > -1:
            # 找到情感原因中的“的”
            d_s, d_e = find_word('的', F, c_e)
            if d_s > -1:
                # 找到情感原因末端
                n_s, c_e = find_word('n', FN, d_e)
                if n_s > -1:
                    C = F[c_s: c_e + 1]
                    # 找到第一个的
                    i_s, i_e = find_word('的', F, e_e, c_s + 1)
                    masks[f][i_e+1:] = 1
                    if i_s > -1:
                        I = '的'
                        return 'F', I, E, C, masks
    return


def rule11(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)
    masks = np.array([np.zeros_like(np.array(clause), dtype=int) for clause in clauses])
    masks[f][:k_s] = 1

    # 找情感原因
    c_s, c_e = rfind_word('nvn', FN, 0, k_s)
    if c_s > -1 and k_s > 3:
        C = F[c_s: c_e + 1]
        # 找情感对象
        e_s, e_e = find_word('n', FN, k_e)
        if e_s > -1:
            E = F[e_s: e_e + 1]
            return 'F', '', E, C, masks
    return


def rule12(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)
    masks = np.array([np.zeros_like(np.array(clause), dtype=int) for clause in clauses])

    cues = markers[2]
    for word in cues:
        # 找线索词
        i_s, i_e = find_word(word, F, k_e, strict=strict)
        masks[f][i_e + 1:] = 1
        if i_s > -1:
            I = F[i_s: i_e + 1]
            # 找情感原因
            c_s, c_e = find_word('nvn', FN, i_e)
            if c_s > -1:
                C = F[c_s: c_e + 1]
                # 找情感对象
                e_s, e_e = rfind_word('n', FN, 0, k_s + 1)
                if e_s > -1:
                    E = F[e_s: e_e + 1]
                    return 'F', I, E, C, masks
    return


def rule13(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)
    masks = np.array([np.zeros_like(np.array(clause), dtype=int) for clause in clauses])

    cues = markers[2]
    for word in cues:
        # 找线索词
        i_s, i_e = find_word(word, B, strict=strict)
        if i_s > -1:
            masks[b][i_e + 1:] = 1
            I = B[i_s: i_e + 1]
            # 找情感原因
            c_s, c_e = find_word('v', BN, i_e)
            if c_s > -1 and len(B) - i_e > 3:
                C = B[c_s: c_e + 1]
                # 找情感对象
                e_s, e_e = rfind_word('n', FN, 0, k_s + 1)
                if e_s > -1:
                    E = F[e_s: e_e + 1]
                    return 'B', I, E, C, masks
    return


def rule14(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)
    masks = np.array([np.zeros_like(np.array(clause), dtype=int) for clause in clauses])

    # 找情感对象
    e_s, e_e = find_word('n', FN, 0, k_s + 1)
    if e_s > -1:
        E = F[e_s: e_e + 1]
        # 找情感原因
        c_s, c_e = rfind_word('v', BN)
        if c_s > -1 and rfind_word('n', BN)[0] > -1:
            masks[b][:] = 1
            C = B[c_s: c_e + 1]
            return 'B', '', E, C, masks
    return


def rule15(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)
    masks = np.array([np.zeros_like(np.array(clause), dtype=int) for clause in clauses])

    # 找情感对象
    e_s, e_e = find_word('n', BN)
    if e_s > -1:
        masks[b][e_e+1:] = 1
        E = B[e_s: e_e + 1]
        # 找情感原因
        c_s, c_e = rfind_word('v', BN, e_e)
        if c_s > -1:
            C = B[c_s: c_e + 1]
            return 'B', '', E, C, masks
    return
