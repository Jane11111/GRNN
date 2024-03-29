# -*- coding: utf-8 -*-
# @Time    : 2021-05-06 20:49
# @Author  : zxl
# @FileName: main_parameter_best.py


import json

import torch as th
from torch.utils.data import DataLoader
from prepare_data.preprocess import PrepareData
from prepare_data.dataset import PaddedDatasetGNN
from prepare_data.collate import collate_gnn_fn
from utils.trainer import TrainRunnerGnn
# from model.grnn_ablation import gated_my_update, wgat_my_update, grnn_no_duplication
# from config.configurator import Config
from model.grnn import GRNN

import yaml
import logging
from logging import getLogger
import os
import numpy as np
import random
import time


def seed_torch(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True


seed_torch()


def load_data(prepare_data_model, model_name,max_len):
    if model_name == 'GRNN_parameter':
        train_sessions, test_sessions, dev_sessions = prepare_data_model.read_train_dev_test_grnn_parameter(max_len)

    prepare_data_model.get_statistics()
    num_items = prepare_data_model.get_item_num()
    train_set = PaddedDatasetGNN(train_sessions, max_len)
    test_set = PaddedDatasetGNN(test_sessions, max_len)
    dev_set = PaddedDatasetGNN(dev_sessions, max_len)
    train_loader = DataLoader(
        train_set,
        batch_size=config['train_batch_size'],
        shuffle=True,
        drop_last=False,
        collate_fn=collate_gnn_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config['train_batch_size'],
        shuffle=False,
        collate_fn=collate_gnn_fn,
        pin_memory=True
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=config['train_batch_size'],
        shuffle=False,
        collate_fn=collate_gnn_fn,
        pin_memory=True
    )
    return num_items, train_loader, test_loader, dev_loader


def load_hyper_param(config, model_name,data_name):
    res = []
    # if model_name == 'GRNN_parameter':
    #     for dropout_prob in [  0.25]:
    #         for gnn_hidden_dropout_prob in [0]:
    #             for gnn_att_dropout_prob in [0]:
    #                 for agg_layer in [  2, 3]:
    #                     for embedding_size in [128]:
    #                         for hidden_size in [128]:
    #                             for lr in [0.001, 0.005, 0.0005]:
    #                                 cur_config = config.copy()
    #                                 cur_config['hidden_size'] = hidden_size
    #                                 cur_config['embedding_size'] = embedding_size
    #                                 cur_config['gnn_hidden_dropout_prob'] = gnn_hidden_dropout_prob
    #                                 cur_config['gnn_att_dropout_prob'] = gnn_att_dropout_prob
    #                                 cur_config['learning_rate'] = lr
    #                                 cur_config['dropout_prob'] = dropout_prob
    #                                 cur_config['agg_layer'] = agg_layer
    #                                 res.append(cur_config)

    embedding_size = 128
    hidden_size = 128
    gnn_hidden_dropout_prob = 0
    gnn_att_dropout_prob = 0
    if data_name == 'elec':
        learning_rate = 0.005
        dropout_prob = 0.25
        agg_layer = 1

    elif data_name == 'tmall_buy':
        learning_rate = 0.001
        dropout_prob = 0
        agg_layer = 1
    elif data_name == 'movielen':
        learning_rate = 0.0005
        dropout_prob = 0.25
        # agg_layer = 3
        agg_layer = 2
    elif data_name == 'movie_tv':
        learning_rate = 0.005
        dropout_prob = 0
        agg_layer = 1
    elif data_name == 'kindle':
        learning_rate = 0.0005
        dropout_prob = 0.25
        agg_layer = 1
    elif data_name == 'home':
        learning_rate = 0.005
        dropout_prob = 0.25
        agg_layer = 3
    elif data_name == 'phone':
        learning_rate = 0.005
        dropout_prob = 0
        agg_layer = 2

    for max_len in [10,20,30,40]:
        # for dropout_prob in [0,0.25]:
        #     for learning_rate in [0.005, 0.001]:
        #         for agg_layer in [1 ]:
        cur_config = config.copy()

        cur_config['hidden_size'] = hidden_size
        cur_config['embedding_size'] = embedding_size
        cur_config['gnn_hidden_dropout_prob'] = gnn_hidden_dropout_prob
        cur_config['gnn_att_dropout_prob'] = gnn_att_dropout_prob
        cur_config['learning_rate'] = learning_rate
        cur_config['dropout_prob'] = dropout_prob
        cur_config['agg_layer'] = agg_layer
        cur_config['max_len'] = max_len
        res.append(cur_config)

    return res


if __name__ == "__main__":

    model = 'GRNN_parameter'
    dataset = 'home'
    gpu_id = 1
    epochs = 300
    train_batch_size = 512
    max_len = 50

    config = {'model': model,
              'dataset': dataset,
              'gpu_id': gpu_id,
              'epochs': epochs,
              'max_len': max_len,
              'train_batch_size': train_batch_size,
              'device': 'cuda:' + str(gpu_id)}

    config_path = './config/' + model + '.yaml'
    with open(config_path, 'r') as f:
        dict = yaml.load(f.read(), Loader=yaml.FullLoader)

    for key in dict:
        config[key] = dict[key]
    # Logging

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

    # max_len = config['max_len']

    founded_best_hit_10 = 0
    founded_best_ndcg_10 = 0

    founded_best_test_hit_5, founded_best_test_ndcg_5, founded_best_test_mrr_5, \
    founded_best_test_hit_10, founded_best_test_ndcg_10, founded_best_test_mrr_10, \
    founded_best_test_hit_20, founded_best_test_ndcg_20, founded_best_test_mrr_20, = \
        0, 0, 0, 0, 0, 0, 0, 0, 0

    # TODO 是不是可以从这里循环调参数
    best_config = config.copy()
    config_lst = load_hyper_param(config, model_name=model,data_name=dataset)

    prepare_data_model = PrepareData(config, logger)

    hyper_count = len(config_lst)
    logger.info('[hyper parameter count]: %d' % (hyper_count))
    hyper_number = 0

    best_model_dict = None


    for config in config_lst:
        seed_torch()
        hyper_number += 1


        num_items, train_loader, test_loader, dev_loader = load_data(prepare_data_model, model_name=model,
                                                                 max_len=config['max_len'])





        logger.info(' start training, running parameters:')
        logger.info(config)


        if config['model'] == 'GRNN_parameter':
            model_obj = GRNN(config, num_items)

        device = config['device']
        model_obj = model_obj.to(device)

        runner = TrainRunnerGnn(
            model_obj,
            train_loader,
            test_loader,
            dev_loader,
            device=device,
            lr=config['learning_rate'],
            weight_decay=0,
            logger=logger,
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
            best_model_dict = runner.get_best_model()

            logger.info('------------founded a better result--------------')

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
                    % (founded_best_hit_10, founded_best_ndcg_10, best_hr_10, best_ndcg_10))
        # logger.info('.................for testing modification...................')
        logger.info('[hyper number]: %d/%d' % (hyper_number, hyper_count))
        logger.info('<founded best test> hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                    'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                    'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                    % (founded_best_test_hit_5, founded_best_test_ndcg_5, founded_best_test_mrr_5,
                       founded_best_test_hit_10, founded_best_test_ndcg_10, founded_best_test_mrr_10,
                       founded_best_test_hit_20, founded_best_test_ndcg_20, founded_best_test_mrr_20))
        logger.info('=================finished current search======================')


    def save_load_best_model():

        model_save_path = 'data/model/' + model + '_' + dataset + '_' + str(cur_time) + '.pkl'
        best_result_save_path = 'data/best_result/' + model + '_' + dataset + '_' + str(cur_time) + '.txt'

        # best_model = runner.model

        th.save(best_model_dict, model_save_path)
        loaded_model_dict = th.load(model_save_path)

        if config['model'] == 'GRNN_parameter':
            best_model = GRNN(best_config, num_items)

        best_model = best_model.to(device)
        best_model.load_state_dict(loaded_model_dict)

        runner = TrainRunnerGnn(
            best_model,  # 最好的model
            train_loader,
            test_loader,
            dev_loader,
            device=device,
            lr=config['learning_rate'],
            weight_decay=0,
            logger=logger,
        )

        test_hit_5, test_ndcg_5, test_mrr_5, \
        test_hit_10, test_ndcg_10, test_mrr_10, \
        test_hit_20, test_ndcg_20, test_mrr_20 = \
            runner.evaluate(best_model, runner.test_loader, runner.device)
        logger.info('<loaded best test> hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                    'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                    'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                    % (test_hit_5, test_ndcg_5, test_mrr_5,
                       test_hit_10, test_ndcg_10, test_mrr_10,
                       test_hit_20, test_ndcg_20, test_mrr_20))

        top1_lst = runner.get_top1(best_model)
        print(top1_lst)

        with open(best_result_save_path, 'w') as w:
            w.write(str(top1_lst))
            w.write('\n')
            w.write(json.dumps(best_config, indent=4))
            w.write('\n')
            w.write('<loaded best test> hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                    'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                    'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                    % (test_hit_5, test_ndcg_5, test_mrr_5,
                       test_hit_10, test_ndcg_10, test_mrr_10,
                       test_hit_20, test_ndcg_20, test_mrr_20))
        # save model


    save_load_best_model()
