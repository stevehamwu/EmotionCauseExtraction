#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/13 14:09
# @Author  : Steve Wu
# @Site    : 
# @File    : eval.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import os
import copy

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import metrics
from metrics.ec.ec import ECMetrics
import statistics
from statistics.ec.ec import ECStatistics
import utils
from utils.app.log import Logger
from utils.app.tensorboard import TBLogger
from utils.dataset.ec import ECDataset
from utils.dataloader.ec import ECDataLoader
import models
from models.han.han import *
from models.han.hcn import HierarchicalConvolutionNetwork
from models.han.gru_att import *


def maxS(alist):
    maxScore = 0.0
    maxIndex = -1
    for i in range(len(alist)):
        if alist[i] > maxScore:
            maxScore = alist[i]
            maxIndex = i
    return maxScore, maxIndex


class EvalSession:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        data_root = self.config['_data.data_root']
        self.data_dir = os.path.join(data_root, 'static.{}'.format(self.args['exp']))

        model_root = self.config['_model.model_root']
        if not os.path.exists(model_root):
            return
        self.model_dir = os.path.join(model_root, self.config["_model.name"], self.args['session'])
        if not os.path.exists(self.model_dir):
            return

        eval_root = self.config["_eval.eval_root"]
        if not os.path.exists(eval_root):
            os.makedirs(eval_root)
        self.eval_dir = os.path.join(eval_root, self.config["_model.name"], self.args['session'])
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)

        self.stat_dir = os.path.join(eval_root, self.config["_model.name"])
        if not os.path.exists(self.stat_dir):
            os.makedirs(self.stat_dir)

        session_config = os.path.join(self.eval_dir, self.args['session'] + '.yaml')
        self.save_session_config(session_config)

        log_file = os.path.join(self.eval_dir, self.args['session'] + ".log")
        Logger.init_instance(log_file)
        self.logger = Logger.get_instance()
        self.logger.info('Build model for evaluation')
        self.logger.info("<<<<<<<<[<[new run]>]>>>>>>>>")
        self.logger.info("session config as follow:")
        self.config.print_to(self.logger)
        self.logger.info("now start session.")

        if self.config['_eval.debug_level'] == 0:
            self.logger.info('Debug mode, load test data')
            self.data_dir = '/data/wujipeng/ec/data/test/'

        self.logger.info("load data for training.")
        self.train_dataset = eval(self.config["_data.class"])(train=True, data_root=self.data_dir,
                                                              **self.config["_data.settings"].todict())

        self.logger.info("load data for evaluation")
        self.eval_dataset = eval(self.config["_data.class"])(train=False, data_root=self.data_dir,
                                                             **self.config["_data.settings"].todict())

        dataset = self.train_dataset + self.eval_dataset
        sentences = list(zip(*dataset))[1] + list(zip(*dataset))[2]
        self.sequence_length = max([max([len(clause) for clause in sentence]) for sentence in sentences])

        # Build model
        self.logger.info("build models.")
        self.model = eval(self.config["_model.class"])(name=self.config['_model.name'],
                                                       **self.config['_model.settings'].todict())

        file_name = os.path.join(self.model_dir, self.args['session'] + '.best.pth')
        self.logger.info("loading model weights from {}.".format(file_name))
        self.model.load_state_dict(self.load_by_torch(file_name))

        # Build criterion
        self.logger.info("build criterion.")
        self.criterion = eval(self.config["_eval.criterion.class"])(
            **self.config["_eval.criterion.args"].todict())

        # Build metrics
        self.logger.info("build metrics")
        self.metrics = eval(self.config["_eval.metrics.class"])(self.config["_eval.metrics.args"])

        # Build statistics
        self.logger.info("build statistics")
        self.statistics = eval(self.config["_eval.statistics.class"])()

        if self.config.get("_eval.gpu") is not None:
            self.args["gpu"] = self.config["_eval.gpu"]
        if self.args['gpu'] < 0:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda: {}".format(self.args['gpu']))
        self.logger.info("Evaluate on {}".format(self.device))
        self.model = self.model.to(self.device)
        if 'crf' in self.config["_model.name"]:
            self.model.set_device(self.device)
        self.criterion = self.criterion.to(self.device)

        self.stat = self.init_stat()
        # self.step = 0

    def eval(self):
        total, acc, pre, rec, f1, loss, all_probs, auc = self.evaluate(self.train_dataset)
        pre_ranking, rec_ranking, f1_ranking = self.statistics(self.train_dataset, all_probs, self.eval_dir,
                                                               'train.best')
        self.logger.info('****************************************')
        self.logger.info('-----------------------')
        self.logger.info('The session {}'.format(self.args["session"]))
        self.logger.info('Train Loss: {:.6f}'.format(loss))
        self.logger.info('Train Accuracy: {:.6f}'.format(acc))
        self.logger.info('Train Roc auc: {:.6f}'.format(auc))
        self.logger.info('Train Precision: {:.6f}'.format(pre))
        self.logger.info('Train Recall: {:.6f}'.format(rec))
        self.logger.info('Train F1: {:.6f}'.format(f1))
        self.logger.info('-----------------------')
        self.logger.info('Train Ranking Precision: {:.6f}'.format(pre_ranking))
        self.logger.info('Train Ranking Recall: {:.6f}'.format(rec_ranking))
        self.logger.info('Train Ranking F1: {:.6f}'.format(f1_ranking))

        total, acc, pre, rec, f1, loss, all_probs, auc = self.evaluate(self.eval_dataset)
        pre_ranking, rec_ranking, f1_ranking = self.statistics(self.eval_dataset, all_probs, self.eval_dir, 'best')
        self.logger.info('****************************************')
        self.logger.info('-----------------------')
        self.logger.info('The session {}'.format(self.args["session"]))
        self.logger.info('Eval Loss: {:.6f}'.format(loss))
        self.logger.info('Eval Accuracy: {:.6f}'.format(acc))
        self.logger.info('Eval Roc auc: {:.6f}'.format(auc))
        self.logger.info('Eval Precision: {:.6f}'.format(pre))
        self.logger.info('Eval Recall: {:.6f}'.format(rec))
        self.logger.info('Eval F1: {:.6f}'.format(f1))
        self.logger.info('-----------------------')
        self.logger.info('Eval Ranking Precision: {:.6f}'.format(pre_ranking))
        self.logger.info('Eval Ranking Recall: {:.6f}'.format(rec_ranking))
        self.logger.info('Eval Ranking F1: {:.6f}'.format(f1_ranking))
        self.update_stat(acc, pre, rec, f1, pre_ranking, rec_ranking, f1_ranking)

        self.logger.info("save final stat")
        self.save_stat()
        self.logger.info("final stat saved.")

    def evaluate(self, dataset):
        self.val_loader = ECDataLoader(dataset=dataset, clause_length=self.sequence_length,
                                       batch_size=self.config['_eval.batch_size'],
                                       shuffle=False, sort=False, collate_fn=self.eval_dataset.collate_fn)
        self.model.eval()
        losses = 0.
        all_probs, all_preds, all_targets = [], [], []
        all_clauses = []
        for batch in self.val_loader:
            clauses, keywords, poses = dataset.batch2input(batch)
            labels = dataset.batch2target(batch)
            clauses = torch.from_numpy(clauses).to(self.device)
            keywords = torch.from_numpy(keywords).to(self.device)
            poses = torch.from_numpy(poses).to(self.device)
            labels = torch.from_numpy(labels).to(self.device)

            if not self.config.get("_train.multi_loss") or not self.config["_train.multi_loss"]:
                if not self.config.get("_train.inline_loss") or not self.config["_train.inline_loss"]:
                    probs, word_attn, sentence_attn = self.model(clauses, keywords, poses)
                    predicts = probs.max(dim=-1)[1]

                    loss = self.criterion(probs.view(-1, probs.size(-1)), labels.view(-1)).item()
                elif self.config["_train.inline_loss"]:
                    loss, predicts, probs, word_attn, sentence_attn = self.model(clauses, keywords, poses)
                    loss = loss.item()

            elif self.config["_train.multi_loss"]:
                word_probs, probs, word_attn, sentence_attn = self.model(clauses, keywords, poses)
                predicts = probs.max(dim=-1)[1]
                loss1 = self.criterion(word_probs.view(-1, probs.size(-1)), labels.view(-1))
                loss2 = self.criterion(probs.view(-1, probs.size(-1)), labels.view(-1))
                coef = self.config["_train.loss_rate"]
                loss = coef[0] * loss1 + coef[1] * loss2
            losses += loss

            probs = F.softmax(probs, dim=-1)

            all_clauses += clauses.tolist()
            all_probs += probs.tolist()
            all_preds += predicts.tolist()
            all_targets += labels.tolist()

        total, acc, pre, rec, f1, auc = self.metrics(all_preds, all_targets, np.concatenate(all_probs)[:, 1])
        return total, acc, pre, rec, f1, losses, all_probs, auc

    def load_by_torch(self, file_name):
        """
        Note: pre-assumed the existence of file_name.
        """
        # file manipulation
        return torch.load(file_name)

    def save_session_config(self, file_name):
        if os.path.exists(file_name):
            old_backup_file = file_name + ".backup"
            if os.path.exists(old_backup_file):
                os.remove(old_backup_file)
            os.rename(file_name, old_backup_file)
        self.config.to_file(file_name)

    def init_stat(self):
        stat = {
            'acc_list': [],
            'pre_list': [],
            'rec_list': [],
            'f1_list': [],
            'pre_ranking_list': [],
            'rec_ranking_list': [],
            'f1_ranking_list': [],
        }
        if os.path.exists(
                os.path.join(self.stat_dir,
                             (self.config['_model.name'] if self.config["_train.debug_level"] > 0 else 'test')
                             + '.stat')):
            stat['stat_all'] = json.load(open(
                os.path.join(self.stat_dir, (
                    self.config['_model.name'] if self.config["_train.debug_level"] > 0 else 'test') + '.stat'), 'r'))
        else:
            stat['stat_all'] = {
                'acc_all': 0.0,
                'pre_all': 0.0,
                'rec_all': 0.0,
                'f1_all': 0.0,
                'pre_all_ranking': 0.0,
                'rec_all_ranking': 0.0,
                'f1_all_ranking': 0.0,
                'acc_all_list': [],
                'pre_all_list': [],
                'rec_all_list': [],
                'f1_all_list': [],
                'pre_all_ranking_list': [],
                'rec_all_ranking_list': [],
                'f1_all_ranking_list': [],
                'time_list': []
            }
        return stat

    def update_stat(self, acc, pre, rec, f1, pre_ranking, rec_ranking, f1_ranking):
        self.stat['acc_list'].append(acc)
        self.stat['pre_list'].append(pre)
        self.stat['rec_list'].append(rec)
        self.stat['f1_list'].append(f1)
        self.stat['pre_ranking_list'].append(pre_ranking)
        self.stat['rec_ranking_list'].append(rec_ranking)
        self.stat['f1_ranking_list'].append(f1_ranking)

    def disp_stat(self, max_index, max_score):
        self.logger.info('#########################################')
        self.logger.info('The time {}'.format(self.args['curr_time']))
        self.logger.info('pre {}'.format(self.stat['pre_list'][max_index]))
        self.logger.info('rec {}'.format(self.stat['rec_list'][max_index]))
        self.logger.info('f1 {}'.format(self.stat['f1_list'][max_index]))
        self.logger.info('pre_ranking {}'.format(self.stat['pre_ranking_list'][max_index]))
        self.logger.info('rec_ranking {}'.format(self.stat['rec_ranking_list'][max_index]))
        self.logger.info('f1_ranking {}'.format(self.stat['f1_ranking_list'][max_index]))
        self.logger.info('#########################################')

    def save_stat(self):
        max_score, max_index = maxS(self.stat['f1_list'])
        self.stat['stat_all']['pre_all'] += self.stat['pre_list'][max_index]
        self.stat['stat_all']['rec_all'] += self.stat['rec_list'][max_index]
        self.stat['stat_all']['f1_all'] += self.stat['f1_list'][max_index]
        self.stat['stat_all']['pre_all_ranking'] += self.stat['pre_ranking_list'][max_index]
        self.stat['stat_all']['rec_all_ranking'] += self.stat['rec_ranking_list'][max_index]
        self.stat['stat_all']['f1_all_ranking'] += self.stat['f1_ranking_list'][max_index]
        self.stat['stat_all']['pre_all_list'].append(self.stat['pre_list'][max_index])
        self.stat['stat_all']['rec_all_list'].append(self.stat['rec_list'][max_index])
        self.stat['stat_all']['f1_all_list'].append(self.stat['f1_list'][max_index])
        self.stat['stat_all']['pre_all_ranking_list'].append(self.stat['pre_ranking_list'][max_index])
        self.stat['stat_all']['rec_all_ranking_list'].append(self.stat['rec_ranking_list'][max_index])
        self.stat['stat_all']['f1_all_ranking_list'].append(self.stat['f1_ranking_list'][max_index])
        self.stat['stat_all']['time_list'].append(self.args['curr_time'])
        self.disp_stat(max_index, max_score)

        json.dump(self.stat['stat_all'],
                  open(os.path.join(self.stat_dir,
                                    (self.config['_model.name']
                                     if self.config["_train.debug_level"] > 0 else 'test') + '.stat'),
                       'w'), ensure_ascii=False, indent=4)
