#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/29/2023 4:39 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : main.py
# @Software: PyCharm
import datetime
import logging
import sys
from pathlib import Path
import time

import numpy as np
import torch

torch.cuda.is_available()
import os
# print(torch.cuda.get_device_name(1))
from tensorboardX import SummaryWriter
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'data'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from lib.dbscan_utils import validate
from dataset.dataloader import data_loader
from utils.parse import parse_args, train


if __name__ == '__main__':
    config = parse_args()
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['util']['gpu'])
    '''CREATE DIR'''
    if config['training']['log_dir'] == 'DGCNN' and config['model']['md'] != 'dg':
        raise Exception(f"Model backbone {config['model']['md']} and log directory {config['training']['log_dir']} does not match")
    if config['training']['log_dir'] == 'PointNet' and config['model']['md'] != 'pn':
        raise Exception(f"Model backbone {config['model']['md']} and log directory {config['training']['log_dir']} does not match")
    times = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path(f"./log/{config['training']['log_dir']}/" + times)
    os.makedirs(experiment_dir,exist_ok=True)
    # experiment_dir.mkdir(exist_ok=True)
    exp = 'sinkhorn'
    experiment_dir = experiment_dir.joinpath(exp)
    experiment_dir.mkdir(exist_ok=True)
    if config['training']['log_dir'] is None:
        experiment_dir = experiment_dir.joinpath(times)
    else:
        experiment_dir = experiment_dir.joinpath(config['training']['log_dir'])
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, config['model']['name']))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    '''DATA LOADING'''
    logger.info(config)
    logger.info('Load dataset ...')
    train_data, val_data = data_loader(config)
    train(config, logger, train_data, val_data, checkpoints_dir,exp)
