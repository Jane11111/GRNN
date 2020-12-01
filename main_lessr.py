# -*- coding: utf-8 -*-
# @Time    : 2020-11-15 11:59
# @Author  : zxl
# @FileName: main.py

import argparse

import warnings

warnings.filterwarnings('ignore')

from pathlib import Path
import torch as th
from torch.utils.data import DataLoader
from utils.data.preprocess import PrepareData
from utils.data.dataset import DatasetLessr
from utils.data.collate import collate_fn_lessr
from utils.trainer import TrainRunnerLessr

from model.baseline.lessr_fast import LESSR_fast
# from config.configurator import Config
import yaml
import logging
from logging import getLogger

import time
import os
import numpy as np
import random

def seed_torch(seed=2020):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	th.manual_seed(seed)
	th.cuda.manual_seed(seed)
	th.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	th.backends.cudnn.benchmark = False
	th.backends.cudnn.deterministic = True

seed_torch()

def load_data(prepare_data_model):
    train_sessions, test_sessions, dev_sessions = prepare_data_model.read_train_dev_test_normal()
    prepare_data_model.get_statistics()
    num_items = prepare_data_model.get_item_num()
    train_set = DatasetLessr(train_sessions, max_len)
    test_set = DatasetLessr(test_sessions, max_len)
    dev_set = DatasetLessr(dev_sessions, max_len)
    train_loader = DataLoader(
        train_set,
        batch_size=config['train_batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=4,
        collate_fn=collate_fn_lessr,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config['train_batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_lessr,
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=config['train_batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_lessr,
    )

    return num_items, train_loader, test_loader, dev_loader


def load_hyper_param(config,model):
    config_path = './data/config/model/' + model + '/config.yaml'
    with open(config_path, 'r') as f:
        dict = yaml.load(f.read(), Loader=yaml.FullLoader)




if __name__ == "__main__":

    model = 'LESSR_fast'
    dataset = 'fs'
    gpu_id = 0
    epochs = 300
    train_batch_size = 512



    config = {'model': model,
              'dataset': dataset,
              'gpu_id': gpu_id,
              'epochs': epochs,
              'max_len':50,
              'train_batch_size':train_batch_size,
              'device': 'cuda:'+str(gpu_id) }

    th.cuda.set_device(th.device(config['device']))

    config_path = './data/config/model/'+model+'/config.yaml'
    with open(config_path, 'r') as f:
        dict = yaml.load(f.read(),Loader=yaml.FullLoader)

    for key in dict:
        config[key] = dict[key]

    cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_path = './data/log/' + config['model'] + '_' + config['dataset'] + '-' + str(cur_time) + "_log.txt"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = getLogger('test')
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s   %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(log_path)
    logger.info(config)

    max_len = config['max_len']


    prepare_data_model = PrepareData(config,logger)
    num_items,train_loader, test_loader, dev_loader = load_data(prepare_data_model)

    founded_best_hit_10 = 0
    founded_best_ndcg_10 = 0

    founded_best_test_hit_5, founded_best_test_ndcg_5, founded_best_test_mrr_5, \
    founded_best_test_hit_10, founded_best_test_ndcg_10, founded_best_test_mrr_10, \
    founded_best_test_hit_20, founded_best_test_ndcg_20, founded_best_test_mrr_20, =\
    0,0,0,0,0,0,0,0,0

    # TODO 是不是可以从这里循环调参数
    best_config = config.copy()

    """
    # for dropout in [0, 0.25, 0.5]:# GRU4Rec
    # for dropout_probs in [[0,0],[0,0.25],[0,0.5],[0.25,0.25],[0.25,0.5],[0.5,0.5]]:# NARM
        config['learning_rate'] = lr
        # config['dropout_prob'] = dropout # GRU4Rec
        # config['dropout_probs'] = dropout_probs# NARM

    """
    # TODO SRGNN 模型还有一个step = 4 需要运行

    # TODO taobaoapp lessr模型还差一个0.0005的需要运行

    for lr in [0.001,  0.005, 0.0001, 0.0005]:
        for step in [1,2,3,4]:
            config['learning_rate'] = lr
            config['step'] = step # SRGNN, GCSAN, LESSR

            logger.info(' start training, running parameters:')
            logger.info(config)

            if config['model'] == 'LESSR':
                model = LESSR_fast(config, num_items)
            device = config['device']


            model = model.to(device)

            runner = TrainRunnerLessr(
                model,
                train_loader,
                test_loader,
                dev_loader,
                device=device,
                lr=config['learning_rate'],
                weight_decay=0,
                logger = logger,
            )

            best_test_hit_5, best_test_ndcg_5, best_test_mrr_5, \
            best_test_hit_10, best_test_ndcg_10, best_test_mrr_10, \
            best_test_hit_20, best_test_ndcg_20, best_test_mrr_20, \
            best_hr_10, best_ndcg_10=\
            runner.train(config['epochs'])



            if best_hr_10 > founded_best_hit_10 and best_ndcg_10> founded_best_ndcg_10:
                founded_best_hit_10 = best_hr_10
                founded_best_ndcg_10 = best_ndcg_10
                best_config = config.copy()

                founded_best_test_hit_5, founded_best_test_ndcg_5, founded_best_test_mrr_5, \
                founded_best_test_hit_10, founded_best_test_ndcg_10, founded_best_test_mrr_10, \
                founded_best_test_hit_20, founded_best_test_ndcg_20, founded_best_test_mrr_20, = \
                best_test_hit_5, best_test_ndcg_5, best_test_mrr_5,\
                best_test_hit_10, best_test_ndcg_10, best_test_mrr_10,\
                best_test_hit_20, best_test_ndcg_20, best_test_mrr_20
            logger.info('finished')
            logger.info('[current config]')
            logger.info(config)
            logger.info('[best config]')
            logger.info(best_config)
            logger.info(
                '[score]: founded best [hit@10: %.5f, ndcg@10: %.5f], current [hit@10: %.5f, ndcg@10: %.5f]'
                % (founded_best_hit_10, founded_best_ndcg_10, best_hr_10, best_ndcg_10))
            logger.info('<founded best test> hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                        'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                        'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                        % (founded_best_test_hit_5, founded_best_test_ndcg_5, founded_best_test_mrr_5,
                            founded_best_test_hit_10, founded_best_test_ndcg_10, founded_best_test_mrr_10,
                            founded_best_test_hit_20, founded_best_test_ndcg_20, founded_best_test_mrr_20))
            logger.info('=================finished current search======================')