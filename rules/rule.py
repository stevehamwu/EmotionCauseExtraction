#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/13 20:49
# @Author  : Steve Wu
# @Site    : 
# @File    : rule.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
# params f, clauses, natures, K,
from .util import *

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

    # constraint
    constraint_words = ['的', '的是', '是']
    for word in constraint_words:
        c_s, c_e = find_word(word, F, k_e, -1)
        if c_s > -1:
            return

    cues = markers[0]  # I

    for word in cues:
        # 找线索词
        i_s, i_e = find_word(word, F, 0, k_s + 1, strict=strict)
        if i_s > -1:
            I = word
            # 找情感对象
            e_s, e_e = find_word('n', FN, i_e, k_s + 1)
            if e_s > -1:
                E = F[e_s: e_e + 1]
                # 在当前句找情感原因
                c_s, c_e = rfind_word('nvn', FN, 0, i_s + 1)
                if c_s > -1:
                    C = F[c_s: c_e + 1]
                    return ('F', I, E, C)
                else:
                    # 在前一句找情感原因
                    c_s, c_e = rfind_word('nvn', BN)
                    if c_s > -1:
                        C = B[c_s: c_e + 1]
                        return ('B', I, E, C)
    return


# params f, clauses, natures, K,
def rule2(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)

    cues = markers[1] + markers[3] + markers[4] + markers[5]  # I
    # constraint
    cues.remove('有')
    for word in cues:
        # 在当前句找线索词
        i_s, i_e = find_word(word, F, 0, k_s + 1, strict=strict)
        if i_s > -1:
            I = F[i_s: i_e + 1]
            # 找情感原因
            c_s, c_e = find_word('nvn', FN, i_e, k_s + 1)
            if c_s > -1:
                C = F[c_s: c_e + 1]
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
                    return ('F', I, E, C)
        else:
            # 在前一句找线索词
            i_s, i_e = find_word(word, B, strict=True)
            if i_s > -1:
                I = B[i_s: i_e + 1]
                # 找情感对象
                e_s, e_e = rfind_word('n', BN, 0, i_s + 1)
                if e_s > -1:
                    E = B[e_s: e_e + 1]
                    # 在前一句找情感原因
                    c_s, c_e = find_word('nvn', BN, i_e)
                    if c_s > -1:
                        C = B[c_s: c_e + 1]
                        return ('B', I, E, C)
                    else:
                        # 在当前句找情感原因
                        c_s, c_e = find_word('nvn', FN, 0, k_s + 1)
                        if c_s > -1:
                            C = F[c_s: c_e + 1]
                            return ('F', I, E, C)
    return


# params f, clauses, natures, K,
def rule3(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)

    cues = markers[1] + markers[3] + markers[4] + markers[5]  # I

    for word in cues:
        # 找线索词
        i_s, i_e = find_word(word, B, strict=strict)
        if i_s > -1:
            I = B[i_s: i_e + 1]
            # 找情感原因
            c_s, c_e = find_word('nvn', BN)
            if c_s > -1:
                C = B[c_s: c_e + 1]
                # 找情感对象
                e_s, e_e = find_word('n', FN, 0, k_s + 1)
                E = F[e_s: e_e + 1]
                if e_s > -1:
                    return ('B', I, E, C)
    return


# params f, clauses, natures, K,
def rule4(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)

    cues = markers[3] + markers[6]

    for word in cues:
        # 找线索词
        i_s, i_e = find_word(word, F, k_e, strict=strict)
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
                c_s, c_e = find_word('nvn', FN, i_e)
                if c_s > -1:
                    C = F[c_s: c_e + 1]
                    return ('F', I, E, C)
                else:
                    # 在下一句找情感原因
                    c_s, c_e = find_word('nvn', AN)
                    if c_s > -1:
                        C = A[c_s: c_e + 1]
                        return ('A', I, E, C)
    return


# params f, clauses, natures, K,
def rule5(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)

    cues = markers[5]

    for word in cues:
        # 找线索词
        i_s, i_e = find_word(word, A, strict=strict)
        if i_s > -1:
            I = A[i_s: i_e + 1]
            # 找情感对象
            e_s, e_e = find_word('n', FN, 0, k_s + 1)
            if e_s > -1:
                E = F[e_s: e_e + 1]
                # 找情感原因
                c_s, c_e = find_word('nvn', AN, i_e)
                if c_s > -1:
                    C = A[c_s: c_e + 1]
                    return ('A', I, E, C)
    return


# params f, clauses, natures, K,
def rule6(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)

    cues = markers[0]

    for word in cues:
        # 找线索词
        i_s, i_e = find_word(word, F, strict=strict)
        if i_s > -1:
            I = F[i_s: i_e + 1]
            # 找情感对象
            e_s, e_e = find_word('n', FN, i_e, k_s + 1)
            if e_s > -1:
                E = F[e_s: e_e + 1]
                # 在当前句找情感原因
                c_s, c_e = find_word('nvn', FN, k_e)
                if c_s > -1:
                    C = F[c_s: c_e + 1]
                    return ('F', I, E, C)
                else:
                    # 在下一句找情感原因
                    c_s, c_e = find_word('nvn', AN)
                    if c_s > -1:
                        C = A[c_s: c_e + 1]
                        return ('A', I, E, C)
    return


# params f, clauses, natures, K,
def rule7(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)

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
                    return ('F', I, E, C)


def rule8(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)

    # 找情感对象
    e_s, e_e = find_word('n', FN, 0, k_s + 1)
    if e_s > -1:
        E = F[e_s: e_e + 1]
        # 找情感原因
        c_s, c_e = find_word('nvn', FN, k_e)
        if c_s > -1:
            C = F[c_s: c_e + 1]
            return ('F', '', E, C)
    return


def rule9(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)

    cues = markers[3]
    for word in cues:
        # 找线索词
        i_s, i_e = rfind_word(word, F, 0, k_s + 1, strict=strict)
        if i_s > -1 and k_s - i_e > 1:
            I = F[i_s: i_e + 1]
            C = F[i_s: k_s]
            # 找情感对象
            e_s, e_e = rfind_word('n', FN, 0, i_s + 1)
            if e_s > -1:
                E = F[e_s: e_e + 1]
                return ('F', I, E, C)
    return


def rule10(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)

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
                    if i_s > -1:
                        I = '的'
                        return ('F', I, E, C)
    return


def rule11(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)

    # 找情感原因
    c_s, c_e = rfind_word('nvn', FN, 0, k_s)
    if c_s > -1:
        C = F[c_s: c_e + 1]
        # 找情感对象
        e_s, e_e = find_word('n', FN, k_e)
        if e_s > -1:
            E = F[e_s: e_e + 1]
            return ('F', '', E, C)
    return


def rule12(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)

    cues = markers[2]
    for word in cues:
        # 找线索词
        i_s, i_e = find_word(word, F, k_e, strict=strict)
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
                    return ('F', I, E, C)
    return


def rule13(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)

    cues = markers[2]
    for word in cues:
        # 找线索词
        i_s, i_e = find_word(word, B, strict=strict)
        if i_s > -1:
            I = B[i_s: i_e + 1]
            # 找情感原因
            c_s, c_e = find_word('nvn', BN, i_e)
            if c_s > -1:
                C = B[c_s: c_e + 1]
                # 找情感对象
                e_s, e_e = rfind_word('n', FN, 0, k_s + 1)
                if e_s > -1:
                    E = F[e_s: e_e + 1]
                    return ('B', I, E, C)
    return


def rule14(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)

    # 找情感对象
    e_s, e_e = find_word('n', FN, 0, k_s + 1)
    if e_s > -1:
        E = F[e_s: e_e + 1]
        # 找情感原因
        c_s, c_e = rfind_word('nvn', BN)
        if c_s > -1:
            C = B[c_s: c_e + 1]
            return ('B', '', E, C)
    return


def rule15(f, clauses, natures, K, strict=True):
    b, a = f - 1, f + 1
    B, F, A = clauses[b] if b >= 0 else None, clauses[f], clauses[a] if a < len(clauses) else None
    BN, FN, AN = natures[b] if b >= 0 else None, natures[f], natures[a] if a < len(natures) else None
    k_s, k_e = find_word(K, F)

    # 找情感对象
    e_s, e_e = find_word('n', BN)
    if e_s > -1:
        E = B[e_s: e_e + 1]
        # 找情感原因
        c_s, c_e = rfind_word('nvn', BN, e_e)
        if c_s > -1:
            C = B[c_s: c_e + 1]
            return ('B', '', E, C)