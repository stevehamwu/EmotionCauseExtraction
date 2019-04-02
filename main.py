#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/17 11:27
# @Author  : Steve Wu
# @Site    : 
# @File    : main.py
# @Software: PyCharm
# @Github  : https://github.com/stevehamwu
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from train import TrainSession
from eval import EvalSession


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--session", type=str, required=True,
                        help="current session name, distinguishing models between different hparams.(no suffix)")
    parser.add_argument("--exp", type=str, required=True,
                        help="current session exp, distinguishing models between different hparams.(no suffix)")
    parser.add_argument("--config", type=str, required=True,
                        help="config file in specific format. (leave empty to use default)")
    parser.add_argument("--curr_time", type=str, required=True,
                        help="current time of training")
    parser.add_argument("--ckpt_in", type=str,
                        help="checkpoint to be loaded from.(no prefix, no extension)")
    parser.add_argument("--ckpt_out", type=str,
                        help="checkpint to be saved to.(no prefix, no extension)")
    parser.add_argument("--model_in", type=str,
                        help="model_file to be loaded from.(no prefix, no extension)")
    parser.add_argument("--model_out", type=str,
                        help="model_file to be saved to.(no prefix, no extension)")
    parser.add_argument("--result", type=str,
                        help="file_name used to save prediction results.(with no prefix)")
    parser.add_argument("--gpu", type=int, default=2,
                        help="which gpu to use")
    parser.add_argument("--mode", type=str, default="train",
                        help="mode train/eval")

    return vars(parser.parse_args())


def validate_args(args):
    # unsure status of target workspace, ckpt and models can't be verified yet.
    if args["config"] is not None:
        assert os.path.exists(args["config"]), "specified config file doesn't exists."


def main(args):
    config = Config()
    config.from_file(args["config"])
    if args["mode"] == "train":
        session = TrainSession(args, config)
        session.train()
    elif args["mode"] == "eval":
        session = EvalSession(args, config)
        session.eval()


if __name__ == '__main__':
    args = parse_args()
    validate_args(args)
    for key, value in args.items():
        print(key, value)
    main(args)
