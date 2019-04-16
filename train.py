#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/22 17:22
# @Author  : Steve Wu
# @Site    : 
# @File    : train.py
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
from metrics.ec.memnet import MECMetrics
import statistics
from statistics.ec.ec import ECStatistics
from statistics.ec.memnet import MECStatistics
import utils
from utils.app.log import Logger
from utils.app.tensorboard import TBLogger
from utils.dataset.ec import ECDataset
from utils.dataset.rule import ECRuleDataset
from utils.dataset.memnet import MECDataset
from utils.dataset.elmo import ELMoECDataset
from utils.dataloader.ec import ECDataLoader
from utils.dataloader.memnet import MECDataLoader
import models
from models.memnet.memnet import *
from models.han.han import *
from models.han.han_rule import *
from models.han.hcn import HierarchicalConvolutionNetwork
from models.han.gru_att import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def rule_metrics(true_labels, pred_labels, ignore_index=-100):
    if ignore_index in true_labels:
        idx = true_labels.index(ignore_index)
        true_labels = true_labels[: idx]
        pred_labels = pred_labels[: idx]
    acc = accuracy_score(true_labels, pred_labels)
    pre = precision_score(true_labels, pred_labels)
    rec = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    return acc, pre, rec, f1


def maxS(alist):
    maxScore = 0.0
    maxIndex = -1
    for i in range(len(alist)):
        if alist[i] > maxScore:
            maxScore = alist[i]
            maxIndex = i
    return maxScore, maxIndex


class TrainSession:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        data_root = self.config['_data.data_root']
        self.data_dir = os.path.join(data_root, 'static.{}'.format(self.args['exp']))

        model_root = self.config['_model.model_root']
        if not os.path.exists(model_root):
            os.makedirs(model_root)
        self.model_dir = os.path.join(model_root, self.config["_model.name"], self.args['session'])
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.stat_dir = os.path.join(model_root, self.config["_model.name"])
        if not os.path.exists(self.stat_dir):
            os.makedirs(self.stat_dir)

        session_config = os.path.join(self.model_dir, self.args['session'] + '.yaml')
        self.save_session_config(session_config)

        log_file = os.path.join(self.model_dir, self.args['session'] + ".log")
        Logger.init_instance(log_file)
        self.logger = Logger.get_instance()
        self.logger.info("<<<<<<<<[<[new run]>]>>>>>>>>")
        self.logger.info("session config as follow:")
        self.config.print_to(self.logger)
        self.logger.info("now start session.")

        self.tensorboardlogger = TBLogger(os.path.join(self.model_dir, 'logs'))

        if self.config['_train.debug_level'] == 0:
            self.logger.info('Debug mode, load test data')
            if 'memnet' in self.config["_model.name"].lower():
                self.data_dir = '/data10T/data/wujipeng/ec/data/ltp_test'
            else:
                self.data_dir = '/data10T/data/wujipeng/ec/data/test/'

        self.logger.info("load data for training.")
        self.train_dataset = eval(self.config["_data.class"])(train=True, data_root=self.data_dir,
                                                              **self.config["_data.settings"].todict())

        self.logger.info("load data for evaluation")
        self.eval_dataset = eval(self.config["_data.class"])(train=False, data_root=self.data_dir,
                                                             **self.config["_data.settings"].todict())

        dataset = self.train_dataset + self.eval_dataset
        if "memnet" in self.config["_model.name"].lower():
            sentences = list(zip(*dataset))[2] + list(zip(*dataset))[3]
            self.sequence_length = max([len(sentence) for sentence in sentences])
        else:
            sentences = list(zip(*dataset))[1] + list(zip(*dataset))[2]
            self.sequence_length = max([max([len(clause) for clause in sentence]) for sentence in sentences])

        # Build model
        self.logger.info("build models.")
        self.model = eval(self.config["_model.class"])(name=self.config['_model.name'],
                                                       **self.config['_model.settings'].todict())
        self.logger.info("{} parameters to optimize.".format(
            sum(p.numel() for p in self.model.parameters())))
        self.logger.info("prepare for training.")

        # Build criterion
        self.logger.info("build criterion.")
        self.criterion = eval(self.config["_train.criterion.class"])(
            **self.config["_train.criterion.args"].todict())

        # Build metrics
        self.logger.info("build metrics")
        self.metrics = eval(config["_train.metrics.class"])(
            config["_train.metrics.args"])

        # Build statistics
        self.logger.info("build statistics")
        self.statistics = eval(config["_train.statistics.class"])()

        if self.config.get("_train.gpu") is not None:
            self.args["gpu"] = self.config["_train.gpu"]
        if self.args['gpu'] < 0:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda: {}".format(self.args['gpu']))
        self.logger.info("Training on {}".format(self.device))
        self.model = self.model.to(self.device)

        if hasattr(self.model, 'set_device'):
            self.model.set_device(self.device)

        self.criterion = self.criterion.to(self.device)

        self.ckpt = self.init_ckpt()
        if self.args["ckpt_in"] is None:
            # load pretrained if trained for the first time
            self.logger.info("load pretrained for training the first time.")
        else:
            self.logger.info("load checkpoint for resumation.")

            # load ckpt file if restored
            ckpt_file = os.path.join(self.model_dir, self.args["ckpt_in"] + ".ckpt")
            if not os.path.exists(ckpt_file):
                msg = ckpt_file + " doesn't exists."
                self.logger.error(msg)
                raise Exception(msg)
            self.unpack_ckpt(self.load_by_torch(ckpt_file))

        if self.config.get("_train.ranking") and self.config["_train.ranking"]:
            self.ranking = True
        else:
            self.ranking = False

        self.stat = self.init_stat()
        self.step = 0

    def evaluate(self):
        if 'memnet' in self.model.name.lower():
            self.eval_loader = MECDataLoader(dataset=self.eval_dataset, memory_size=self.sequence_length,
                                             sequence_size=3, batch_size=self.config["_train.batch_size"],
                                             shuffle=False, collate_fn=self.eval_dataset.collate_fn)
        else:
            self.eval_loader = ECDataLoader(dataset=self.eval_dataset, clause_length=self.sequence_length,
                                            batch_size=self.config['_train.batch_size'],
                                            shuffle=False, sort=False, collate_fn=self.eval_dataset.collate_fn)
        self.model.eval()
        losses = 0.
        all_probs = []
        all_preds = []
        all_targets = []
        all_rule_preds = []
        all_rule_targets = []
        for batch in self.eval_loader:
            inputs = self.eval_dataset.batch2input(batch)
            if len(inputs) == 4:
                clauses, keywords, masks, poses = inputs
                masks = torch.from_numpy(masks).to(self.device)
                poses = torch.from_numpy(poses).to(self.device)
            elif len(inputs) == 3:
                clauses, keywords, poses = inputs
                poses = torch.from_numpy(poses).to(self.device)
            elif len(inputs) == 2:
                clauses, keywords = inputs
            else:
                return
            clauses = torch.from_numpy(clauses).to(self.device)
            keywords = torch.from_numpy(keywords).to(self.device)

            labels = self.eval_dataset.batch2target(batch)
            labels = torch.from_numpy(labels).to(self.device)

            if not self.config.get("_train.multi_loss") or not self.config["_train.multi_loss"]:
                if not self.config.get("_train.inline_loss") or not self.config["_train.inline_loss"]:
                    if len(inputs) == 4:
                        probs, word_attn, sentence_attn = self.model(clauses, keywords, poses, masks)
                    elif len(inputs) == 3:
                        probs, word_attn, sentence_attn = self.model(clauses, keywords, poses)
                    elif len(inputs) == 2:
                        probs = self.model(clauses, keywords)
                    else:
                        return

                    predicts = probs.max(dim=-1)[1]
                    loss = self.criterion(probs.view(-1, probs.size(-1)), labels.view(-1))
                elif self.config["_train.inline_loss"]:
                    loss, predicts, probs, word_attn, sentence_attn = self.model(clauses, keywords, poses)
                    loss = loss.item()

            elif self.config["_train.multi_loss"]:
                if self.model.name == 'han_rule':
                    rule_labels = self.eval_dataset.batch2rule(batch)
                    rule_labels = torch.from_numpy(rule_labels).to(self.device)
                    rule_probs, probs, word_attn, sentence_attn = self.model(clauses, keywords, poses)
                    predicts = probs.max(dim=-1)[1]
                    rule_predicts = rule_probs.max(dim=1)[1]
                    loss1 = self.criterion(rule_probs, rule_labels)
                    loss2 = self.criterion(probs.view(-1, probs.size(-1)), labels.view(-1))
                    coef = self.config["_train.loss_rate"]
                    # coef[0] *= 0.8**self.ckpt["curr_epoch"]
                    # coef[1] = 1 - coef[0]
                    loss = coef[0] * loss1 + coef[1] * loss2
                    all_rule_preds += rule_predicts.tolist()
                    all_rule_targets += rule_labels.tolist()
                else:
                    word_probs, probs, word_attn, sentence_attn = self.model(clauses, keywords, poses)
                    predicts = probs.max(dim=-1)[1]
                    loss1 = self.criterion(word_probs.view(-1, probs.size(-1)), labels.view(-1))
                    loss2 = self.criterion(probs.view(-1, probs.size(-1)), labels.view(-1))
                    coef = self.config["_train.loss_rate"]
                    loss = coef[0] * loss1 + coef[1] * loss2
            losses += loss

            probs = F.softmax(probs, dim=-1)

            all_probs += probs.tolist()
            all_preds += predicts.tolist()
            all_targets += labels.tolist()

        if 'memnet' in self.model.name.lower():
            pos_probs = np.array(all_probs)[:, 1]
        else:
            pos_probs = np.concatenate(all_probs)[:, 1]
        total, acc, pre, rec, f1, auc = self.metrics(all_preds, all_targets, pos_probs)
        return total, acc, pre, rec, f1, losses, all_probs, auc

    def train(self):
        if 'memnet' in self.model.name.lower():
            self.train_loader = MECDataLoader(dataset=self.train_dataset, memory_size=self.sequence_length,
                                              sequence_size=3, batch_size=self.config["_train.batch_size"],
                                              shuffle=True, collate_fn=self.train_dataset.collate_fn)
        else:
            self.train_loader = ECDataLoader(dataset=self.train_dataset, clause_length=self.sequence_length,
                                             batch_size=self.config['_train.batch_size'], shuffle=True,
                                             sort=True, collate_fn=self.train_dataset.collate_fn)

        if self.config['_train.pretrain']:
            self.logger.info("load pretrained for training the first time.")
            embeddings = pickle.load(open(self.config['_train.pretrained_file'], 'rb')).astype(np.float32)
            self.model.init_weights(torch.from_numpy(embeddings).to(self.device))

        self.logger.info("start training models {} early stopping.".format(
            "with" if self.config["_train.early_stopping_rounds"] > 0 else "without"))
        self.decay_round = 0
        self.early_stop = False

        for epoch in range(self.ckpt["curr_epoch"], self.config['_train.max_epoch']):
            accs = []
            sizes = []

            if self.config["_train.finetune"]:
                if epoch == self.config["_train.finetune_round"]:
                    for para in self.model.Embedding.parameters():
                        para.requires_grad = False

            # Build optimizer
            self.logger.info("build optimizer")
            if not self.config.get('_train.optimizer.multi_lr') or not self.config['_train.optimizer.multi_lr']:
                if self.config.get('_train.optimizer.embedding'):
                    embedding_params = list(map(id, self.model.Embedding.parameters()))
                    base_params = filter(lambda p: id(p) not in embedding_params, self.model.parameters())
                    self.optimizer = eval(self.config["_train.optimizer.class"])([
                        {'params': base_params},
                        {'params': self.model.Embedding.parameters(),
                         'lr': self.config["_train.optimizer.embedding.lr"]}],
                        **self.config["_train.optimizer.args"].todict())
                else:
                    self.optimizer = eval(self.config["_train.optimizer.class"])(
                        self.model.parameters(),
                        **self.config["_train.optimizer.args"].todict())
            elif self.config["_train.optimizer.multi_lr"]:
                self.optimizer = eval(self.config["_train.optimizer.class"])(
                    [
                        {'params': [para for name, para in self.model.named_parameters() if
                                    'word' not in name]},
                        {'params': [para for name, para in self.model.named_parameters() if 'word' in name],
                         'lr': self.config["_train.optimizer.args.lr"] * 0.3}
                    ], **self.config["_train.optimizer.args"].todict())

            losses, losses1, losses2 = 0., 0., 0.
            for i, batch in enumerate(self.train_loader):
                self.ckpt["global_count"] += 1
                self.model.train()
                inputs = self.train_dataset.batch2input(batch)
                if len(inputs) == 4:
                    clauses, keywords, masks, poses = inputs
                    masks = torch.from_numpy(masks).to(self.device)
                    poses = torch.from_numpy(poses).to(self.device)
                elif len(inputs) == 3:
                    clauses, keywords, poses = inputs
                    poses = torch.from_numpy(poses).to(self.device)
                elif len(inputs) == 2:
                    clauses, keywords = inputs
                else:
                    return
                clauses = torch.from_numpy(clauses).to(self.device)
                keywords = torch.from_numpy(keywords).to(self.device)

                labels = self.train_dataset.batch2target(batch)
                labels = torch.from_numpy(labels).to(self.device)
                targets = labels.view(-1)

                loss = 0
                self.optimizer.zero_grad()
                if not self.config.get("_train.multi_loss") or not self.config["_train.multi_loss"]:
                    if not self.config.get("_train.inline_loss") or not self.config["_train.inline_loss"]:
                        if len(inputs) == 4:
                            outputs, word_attn, sentence_attn = self.model(clauses, keywords, poses, masks)
                        elif len(inputs) == 3:
                            outputs, word_attn, sentence_attn = self.model(clauses, keywords, poses)
                        elif len(inputs) == 2:
                            outputs = self.model(clauses, keywords)
                        else:
                            return

                        probs = outputs.view(-1, outputs.size(-1))

                        loss = self.criterion(probs, targets)
                        losses += loss.item()

                        loss.backward()
                        probs = F.softmax(probs, dim=-1)
                    elif self.config["_train.inline_loss"]:
                        masks = labels != self.config["_train.ignore_index"]
                        loss, probs, word_attn, sentence_attn = self.model.neg_log_likelihood(clauses, keywords, poses,
                                                                                              labels, masks)

                        loss.backward()

                elif self.config["_train.multi_loss"]:
                    if self.model.name == 'han_rule':
                        rule_labels = self.train_dataset.batch2rule(batch)
                        rule_labels = torch.from_numpy(rule_labels).to(self.device)
                        rule_probs, probs, word_attn, sentence_attn = self.model(clauses, keywords, poses)
                        probs = probs.view(-1, probs.size(-1))
                        loss1 = self.criterion(rule_probs, rule_labels)
                        loss2 = self.criterion(probs, targets)
                        if epoch == 0:
                            coef = self.config["_train.loss_rate"]
                        # coef[0] *= 0.8
                        # coef[1] = 1 - coef[0]
                        loss = coef[0] * loss1 + coef[1] * loss2
                        losses1 += loss1.item()
                        losses2 += loss2.item()
                        losses += loss.item()
                        rule_pred = rule_probs.max(dim=1)[1]
                        rule_acc, rule_pre, rule_rec, rule_f1 = rule_metrics(rule_labels.tolist(), rule_pred.tolist())
                    else:
                        word_outputs, outputs, word_attn, sentence_attn = self.model(clauses, keywords, poses)
                        word_probs = word_outputs.view(-1, word_outputs.size(-1))
                        probs = outputs.view(-1, outputs.size(-1))
                        loss1 = self.criterion(word_probs, targets)
                        loss2 = self.criterion(probs, targets)
                        coef = self.config["_train.loss_rate"]
                        loss = coef[0] * loss1 + coef[1] * loss2

                    loss.backward()

                if self.config["_train.clip_grad_norm"]:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                               self.config["_train.grad_clip_value"])
                else:
                    grad_norm = 0.0

                if hasattr(self.model, 'gradient_noise_and_clip'):
                    grad_norm = self.model.gradient_noise_and_clip(self.model.parameters(), self.device,
                                                                   noise_stddev=1e-3, max_clip=40.0)

                self.optimizer.step()

                predicts = probs.max(dim=1)[1]
                total, acc, pre, rec, f1, auc = self.metrics(predicts.tolist(), targets.tolist(), probs[:, 1].tolist())

                accs.append(acc)
                sizes.append(total)

                if self.ckpt["global_count"] % self.config['_train.disp_freq'] == 0:
                    if self.model.name == 'xxxhan_rule':
                        self.logger.info('E: [{:3}/{}][{:3}/{}] L: {:.2f}|{:.2f}|{:.2f} A: {:.2f} '
                                         'P: {:.2f} R: {:.2f} F: {:.2f} RF: {:.2f} N: {:.2f}'.format(
                            self.ckpt["curr_epoch"],
                            self.config[
                                "_train.max_epoch"],
                            i, len(self.train_loader),
                            losses1 / (i + 1),
                            losses2 / (i + 1),
                            losses / (i + 1), auc, pre,
                            rec, f1, rule_f1,
                            grad_norm))
                    else:
                        self.logger.info('E: [{:3}/{}][{:3}/{}] L: {:.4f} A: {:.4f} '
                                         'P: {:.4f} R: {:.4f} F: {:.4f} N: {:.4f}'.format(self.ckpt["curr_epoch"],
                                                                                          self.config[
                                                                                              "_train.max_epoch"],
                                                                                          i, len(self.train_loader),
                                                                                          losses / (i + 1), auc, pre,
                                                                                          rec,
                                                                                          f1,
                                                                                          grad_norm))
                # tensorboard
                if self.config['_train.debug_level'] == 0:
                    self.update_tensorboard({'train loss': loss.item(), 'train_roc_auc_score': auc, 'train f1': f1})
                    self.step += 1

                if self.ckpt["global_count"] % self.config['_train.eval_freq'] == 0:
                    self.logger.info('Start evaluation...')
                    total, acc, pre, rec, f1, loss, all_probs, auc = self.evaluate()
                    pre_ranking, rec_ranking, f1_ranking = self.statistics(self.eval_dataset, all_probs, self.model_dir,
                                                                           epoch)
                    self.logger.info('****************************************')
                    self.logger.info('-----------------------')
                    self.logger.info('The time {}'.format(self.args['curr_time']))
                    self.logger.info('Epoch [{}/{}]'.format(self.ckpt["curr_epoch"], self.config["_train.max_epoch"]))
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
                    self.update_tensorboard(
                        {'eval loss': loss, 'eval_roc_auc_score': auc, 'eval f1': f1, 'eval ranking f1': f1_ranking},
                        mode='eval')

                    # check early stopping.
                    curr_eval_score = f1_ranking if self.ranking else f1
                    if self.config["_train.early_stopping_rounds"] > 0:
                        self.ckpt["last_eval_scores"] = ([curr_eval_score]
                                                         + self.ckpt["last_eval_scores"][
                                                           :self.config["_train.early_stopping_rounds"] - 1])
                        recent_best_eval_score = np.max(self.ckpt["last_eval_scores"])
                        if recent_best_eval_score < self.ckpt["best_eval_score"]:
                            self.logger.info("early stopping because of ({:.6f} < {:.6f}).".format(
                                recent_best_eval_score, self.ckpt["best_eval_score"]))
                            self.early_stop = True
                            break

                    # check whether better.
                    if (self.ckpt["best_eval_score"] is None or
                            curr_eval_score > self.ckpt["best_eval_score"]):
                        self.ckpt["best_eval_score"] = curr_eval_score
                        self.ckpt["best_eval_model"] = copy.deepcopy(self.model.state_dict())

                        self.logger.info("saving best eval models.")
                        self.save_by_torch(self.ckpt["best_eval_model"],
                                           os.path.join(self.model_dir, self.args["model_out"] + ".best.pth"))
                # periodical checkpoint. (for resumation)
                if self.config["_train.save_freq"] > 0 and self.ckpt["global_count"] % self.config["_train.save_freq"] \
                        == 0:
                    self.logger.info("auto save new checkpoint.")
                    # 没空间了
                    # self.save_by_torch(self.pack_ckpt(),
                    #                    os.path.join(self.model_dir, self.args["ckpt_out"] + '.ckpt'))

            self.ckpt["curr_epoch"] = epoch + 1
            if self.early_stop:
                # incomplete epoch doesn't need to be saved, otherwise there'll be bug.
                if self.config.get("_train.decay_round") and self.config["_train.decay_round"] > 0:
                    self.decay_round += 1
                    if self.decay_round >= self.config["_train.decay_round"]:
                        break
                    self.logger.info('Decay lr with rate {:.2} for {} time'.format(self.config["_train.decay_rate"],
                                                                                   self.decay_round))
                    self.model.load_state_dict(
                        self.load_by_torch(os.path.join(self.model_dir, self.args["model_out"] + ".best.pth")))
                    self.adjust_learning_rate(rate=self.config["_train.decay_rate"])
                    self.ckpt["last_eval_scores"] = [self.ckpt["best_eval_score"]]
                    self.early_stop = False
                else:
                    break

            if self.config["_train.debug_level"] == 0:
                # save epoch checkpoint. (for historic performance analysis)
                self.logger.info("save checkpoint of epoch {}.".format(epoch))
                self.save_by_torch(self.pack_ckpt(), os.path.join(
                    self.model_dir, self.args["ckpt_out"] + ".epo-{}".format(epoch) + '.ckpt'))

        self.logger.info("save final stat")
        self.save_stat()
        self.logger.info("final stat saved.")

        self.logger.info("save final best models")
        self.save_by_torch(self.ckpt["best_eval_model"],
                           os.path.join(self.model_dir, self.args["model_out"] + ".best.pth"))
        self.logger.info("final best models saved.")

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
        max_score, max_index = maxS(self.stat['f1_ranking_list' if self.ranking else 'f1_list'])
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

    def update_tensorboard(self, info, mode='train'):
        for tag, value in info.items():
            self.tensorboardlogger.scalar_summary(tag, value, self.step + 1)

        # for tag, value in self.model.named_parameters():
        #     if value.requires_grad:
        #         tag = tag.replace('.', '/')
        #         self.tensorboardlogger.histo_summary(tag, value.data.cpu().numpy(), self.step + 1)
        # self.tensorboardlogger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), self.step + 1)

    def save_session_config(self, file_name):
        if os.path.exists(file_name):
            old_backup_file = file_name + ".backup"
            if os.path.exists(old_backup_file):
                os.remove(old_backup_file)
            os.rename(file_name, old_backup_file)
        self.config.to_file(file_name)

    def save_by_torch(self, obj, file_name):
        """
            safe saving of obj via backup to avoid overwriting and inconsistency.
        """
        # file manipulation
        if os.path.exists(file_name):
            backup_file = file_name + ".backup"
            if os.path.exists(backup_file):
                os.remove(backup_file)
            os.rename(file_name, backup_file)
        torch.save(obj, file_name)

    def load_by_torch(self, file_name):
        """
        Note: pre-assumed the existence of file_name.
        """
        # file manipulation
        return torch.load(file_name)

    def init_ckpt(self):
        """
            status descriptors of training process sufficient for resumation.
        """
        ckpt = {
            "curr_epoch": 0,
            "best_eval_score": 0.,
            "best_eval_model": None,
            "global_count": 0,
            "last_eval_scores": []
        }
        return ckpt

    def pack_ckpt(self):
        ckpt = copy.copy(self.ckpt)
        # collect images
        if isinstance(self.model, nn.DataParallel):
            ckpt["models"] = self.model.module.state_dict()
        else:
            ckpt["models"] = self.model.state_dict()
        if isinstance(self.optimizer, nn.DataParallel):
            ckpt["optimizer"] = self.optimizer.module.state_dict()
        else:
            ckpt["optimizer"] = self.optimizer.state_dict()
        ckpt["train_loader"] = self.train_loader.state_dict()
        return ckpt

    def unpack_ckpt(self, ckpt):
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(ckpt["models"])
        else:
            self.model.load_state_dict(ckpt["models"])
        if isinstance(self.optimizer, nn.DataParallel):
            self.optimizer.module.load_state_dict(ckpt["optimizer"])
        else:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.train_loader.load_state_dict(ckpt["train_loader"])
        # restore self.ckpt
        for k in self.ckpt.keys():
            self.ckpt[k] = ckpt[k]

    def adjust_learning_rate(self, rate=0.5):
        lr = self.optimizer.defaults['lr'] * rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
