# -*- coding: utf-8 -*-
# @Time    : 2020-11-15 11:59
# @Author  : zxl
# @FileName: main.py

import argparse


from pathlib import Path
import torch as th
from torch.utils.data import DataLoader
from utils.data.preprocess import PrepareData
from utils.data.dataset import PaddedDataset,PaddedDatasetNormal
from utils.data.collate import collate_fn,collate_fn_normal
from utils.trainer import TrainRunner
from model.grnn import GRNN,GRNN_mlp,GRNN_no_order
from model.grnn_v1 import GRNN_v1
from model.baseline.gru4rec import GRU4Rec
# from config.configurator import Config
import yaml
import logging
from logging import getLogger
import os
import numpy as np
import random
import time

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

    train_sessions, test_sessions, dev_sessions = prepare_data_model.read_train_dev_test()
    prepare_data_model.get_statistics()
    num_items = prepare_data_model.get_item_num()
    train_set = PaddedDataset(train_sessions, max_len)
    test_set = PaddedDataset(test_sessions, max_len)
    dev_set = PaddedDataset(dev_sessions, max_len)
    train_loader = DataLoader(
        train_set,
        batch_size=config['train_batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config['train_batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=config['train_batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    return num_items, train_loader, test_loader, dev_loader


if __name__ == "__main__":

    model = 'GRNN'
    dataset = 'movie_tv'
    gpu_id = 3
    epochs = 300
    train_batch_size = 512

    config = {'model': model,
              'dataset': dataset,
              'gpu_id': gpu_id,
              'epochs': epochs,
              'max_len':50,
              'train_batch_size':train_batch_size,
              'device': 'cuda:'+str(gpu_id) }


    config_path = './data/config/model/'+model+'/config.yaml'
    with open(config_path, 'r') as f:
        dict = yaml.load(f.read(),Loader=yaml.FullLoader)

    for key in dict:
        config[key] = dict[key]

    cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_path = './data/log/'+config['model']+'_'+config['dataset']+'-'+str(cur_time)+"_log.txt"

    logging.basicConfig(level = logging.INFO,format = '%(asctime)s %(message)s')
    logger = getLogger('test')
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s   %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(log_path)
    logger.info(config)

    max_len = config['max_len']

    prepare_data_model = PrepareData(config, logger)
    num_items, train_loader, test_loader, dev_loader = load_data(prepare_data_model)

    founded_best_hit_10 = 0
    founded_best_ndcg_10 = 0

    founded_best_test_hit_5, founded_best_test_ndcg_5, founded_best_test_mrr_5, \
    founded_best_test_hit_10, founded_best_test_ndcg_10, founded_best_test_mrr_10, \
    founded_best_test_hit_20, founded_best_test_ndcg_20, founded_best_test_mrr_20, = \
        0, 0, 0, 0, 0, 0, 0, 0, 0

    # TODO 是不是可以从这里循环调参数
    best_config = config.copy()

    for dropout_prob in [0,0.25 ]:
        for gnn_hidden_dropout_prob in [0 ]:
            for gnn_att_dropout_prob in [0 ]:
                for n_layers in [1 ]:
                    for n_heads in [1 ]:
                        for embedding_size in [128 ]:
                            for hidden_size in [128  ]:
                                for lr in [0.005 ,0.001]:
                                    config['hidden_size'] = hidden_size
                                    config['embedding_size'] = embedding_size
                                    config['gnn_hidden_dropout_prob'] = gnn_hidden_dropout_prob
                                    config['gnn_att_dropout_prob'] = gnn_att_dropout_prob
                                    config['n_layers'] = n_layers
                                    config['n_heads'] = n_heads
                                    config['learning_rate'] = lr
                                    config['dropout_prob'] = dropout_prob

                                    logger.info(' start training, running parameters:')
                                    logger.info(config)

                                    if model == 'GRNN':
                                        model = GRNN(config, num_items)
                                    elif model == 'GRNN_mlp':
                                        model = GRNN_mlp(config,num_items)
                                    elif model == 'GRNN_no_order':
                                        model = GRNN_no_order(config,num_items)
                                    elif model == 'GRNN_v1':
                                        model = GRNN_v1(config,num_items)


                                    device = config['device']
                                    model = model.to(device)

                                    runner = TrainRunner(
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
                                    best_hr_10, best_ndcg_10 = \
                                        runner.train(config['epochs'])

                                    if best_hr_10 > founded_best_hit_10 and best_ndcg_10 > founded_best_ndcg_10:
                                        founded_best_hit_10 = best_hr_10
                                        founded_best_ndcg_10 = best_ndcg_10
                                        best_config = config.copy()
                                        print('------------founded a better result--------------')

                                        founded_best_test_hit_5, founded_best_test_ndcg_5, founded_best_test_mrr_5, \
                                        founded_best_test_hit_10, founded_best_test_ndcg_10, founded_best_test_mrr_10, \
                                        founded_best_test_hit_20, founded_best_test_ndcg_20, founded_best_test_mrr_20, = \
                                            best_test_hit_5, best_test_ndcg_5, best_test_mrr_5, \
                                            best_test_hit_10, best_test_ndcg_10, best_test_mrr_10, \
                                            best_test_hit_20, best_test_ndcg_20, best_test_mrr_20
                                    logger.info('finished')
                                    logger.info('[current config]')
                                    logger.info(config)
                                    logger.info('[best config]')
                                    logger.info(best_config)
                                    logger.info('[score]: founded best [hit@10: %.5f, ndcg@10: %.5f], current [hit@10: %.5f, ndcg@10: %.5f]'
                                                %(founded_best_hit_10, founded_best_ndcg_10, best_hr_10, best_ndcg_10))
                                    # logger.info('.................for testing modification...................')
                                    logger.info('<founded best test> hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                                                'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                                                'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                                                % (founded_best_test_hit_5, founded_best_test_ndcg_5, founded_best_test_mrr_5,
                                                   founded_best_test_hit_10, founded_best_test_ndcg_10, founded_best_test_mrr_10,
                                                   founded_best_test_hit_20, founded_best_test_ndcg_20, founded_best_test_mrr_20))
                                    logger.info('=================finished current search======================')
