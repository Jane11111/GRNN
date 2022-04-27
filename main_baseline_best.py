# -*- coding: utf-8 -*-
# @Time    : 2020-12-26 16:01
# @Author  : zxl
# @FileName: main_baseline_best.py

import warnings
import json
import copy

warnings.filterwarnings('ignore')

import torch as th
from torch.utils.data import DataLoader
from prepare_data.preprocess import PrepareData
from prepare_data.dataset import PaddedDatasetNormal
from prepare_data.collate import collate_fn_normal
from utils.trainer import TrainRunnerNormal
from model.baseline.gru4rec import GRU4Rec
from model.baseline.narm import NARM
from model.baseline.sasrec import SASRec
from model.baseline.stamp import STAMP
from model.baseline.srgnn import SRGNN
from model.baseline.gcsan import GCSAN
from model.grnn import GRNN
from model.baseline.nextitnet import NextItNet
# from model.baseline.lessr import LESSR
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

def load_data(prepare_data_model,max_len):
    train_sessions, test_sessions, dev_sessions = prepare_data_model.read_train_dev_test_normal(max_len)
    prepare_data_model.get_statistics()
    num_items = prepare_data_model.get_item_num()

    train_set = PaddedDatasetNormal(train_sessions, max_len)
    test_set = PaddedDatasetNormal(test_sessions, max_len)
    dev_set = PaddedDatasetNormal(dev_sessions, max_len)
    train_loader = DataLoader(
        train_set,
        batch_size=config['train_batch_size'],
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn_normal,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config['train_batch_size'],
        shuffle=False,
        collate_fn=collate_fn_normal,
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=config['train_batch_size'],
        shuffle=False,
        collate_fn=collate_fn_normal,
    )

    return num_items, train_loader, test_loader, dev_loader


def load_hyper_param(config, model_name,data_name=None):
    learning_rate_lst = [0.001, 0.005, 0.0005]
    step_lst = [1, 2, 3, 4]
    n_layers_lst = [2, 3]
    n_heads_lst = [1, 2]
    dropout_prob_lst = [0, 0.25, 0.5]
    narm_dropout_probs = [[0, 0], [0, 0.25], [0, 0.5], [0.25, 0.25],
                          [0.25, 0.5], [0.5, 0.5] ]
    # narm_dropout_probs = [[0.25,0],[0.5,0],[0.5,0.25]]
    block_lst = [1, 3, 5]

    res = []
    if model_name == 'SRGNN' and data_name == 'elec':
        step = 3
        learning_rate  = 0.005
    if model_name == 'SRGNN' and data_name == 'home':
        step = 1
        learning_rate  = 0.001
    if model_name == 'SRGNN' and data_name == 'movie_tv':
        step = 3
        learning_rate  = 0.001
    if model_name == 'SRGNN' and data_name == 'tmall_buy':
        step = 1
        learning_rate  = 0.005

    for max_len in [10,20,30,40]:

        cur_config = config.copy()
        cur_config['step'] = step
        cur_config['learning_rate'] = learning_rate
        cur_config['max_len'] = max_len


        res.append(cur_config)

    # if model_name == 'GRU4Rec' and data_name == 'movie_tv':
    #     """
    #     {'model': 'GRU4Rec', 'dataset': 'movie_tv',
    #      'embedding_size': 128, 'hidden_size': 128,
    #      'num_layers': 1, 'dropout_prob': 0,
    #      'learning_rate': 0.005}
    #     """
    #     new_config = config.copy()
    #     for k,v in {'model': 'GRU4Rec', 'dataset': 'movie_tv',
    #      'embedding_size': 128, 'hidden_size': 128,
    #      'num_layers': 1, 'dropout_prob': 0,
    #      'learning_rate': 0.005}.items():
    #         new_config[k] = v
    #     res.append(new_config)

    # if model == 'GRU4Rec':
    #     # for learning_rate in learning_rate_lst:
    #     #     for dropout_prob in dropout_prob_lst:
    #     #         cur_config = config.copy()
    #     #         cur_config['learning_rate'] = learning_rate
    #     #         cur_config['dropout_prob'] = dropout_prob
    #     #         res.append(cur_config)
    #     cur_config = config.copy()
    #     cur_config['learning_rate'] = 0.005
    #     cur_config['dropout_prob'] = 0.25
    #     res.append(cur_config)
    # elif model == 'NARM':
    #     for learning_rate in learning_rate_lst:
    #         for dropout_probs in narm_dropout_probs:
    #             cur_config = config.copy()
    #             cur_config['learning_rate'] = learning_rate
    #             cur_config['dropout_probs'] = dropout_probs
    #             res.append(cur_config)
    #
    # elif model == 'STAMP':
    #     for learning_rate in learning_rate_lst:
    #         cur_config = config.copy()
    #         cur_config['learning_rate'] = learning_rate
    #         res.append(cur_config)
    #
    # elif model == 'SASRec':
    #     for learning_rate in [0.001, 0.0005]:
    #         for n_layers in [2, 3]:
    #             for n_heads in [1, 2]:
    #                 for hidden_dropout_prob in [0, 0.25]:
    #                     for attn_dropout_prob in [0, 0.25]:
    #                         cur_config = config.copy()
    #                         cur_config['learning_rate'] = learning_rate
    #                         cur_config['n_layers'] = n_layers
    #                         cur_config['n_heads'] = n_heads
    #                         cur_config['hidden_dropout_prob'] = hidden_dropout_prob
    #                         cur_config['attn_dropout_prob'] = attn_dropout_prob
    #                         res.append(cur_config)
    # elif model == 'SRGNN':
    #     for learning_rate in learning_rate_lst:
    #         for step in step_lst:
    #             cur_config = config.copy()
    #             cur_config['learning_rate'] = learning_rate
    #             cur_config['step'] = step
    #             res.append(cur_config)
    # elif model == 'GCSAN':
    #     for learning_rate in [ 0.0005]:
    #         for n_layers in [2 ]:
    #             for n_heads in [1 ]:
    #                 for hidden_dropout_prob in [ 0.25 ]:
    #                     for attn_dropout_prob in [0,0.25 ]:
    #                         for step in [1,2,3 ]:
    #                             cur_config = config.copy()
    #                             cur_config['learning_rate'] = learning_rate
    #                             cur_config['n_layers'] = n_layers
    #                             cur_config['n_heads'] = n_heads
    #                             cur_config['hidden_dropout_prob'] = hidden_dropout_prob
    #                             cur_config['attn_dropout_prob'] = attn_dropout_prob
    #                             cur_config['step'] = step
    #                             res.append(cur_config)
    # # elif model == 'GCSAN':
    # #     for step in [2,3,4]:
    # #         cur_config = config.copy()
    # #         cur_config['step'] = step
    # #         if config['dataset'] == 'tmall_buy':
    # #             learning_rate = 0.001
    # #             n_layers = 2
    # #             n_heads = 2
    # #             hidden_dropout_prob = 0.25
    # #             attn_dropout_prob = 0
    # #         elif config['dataset'] == 'elec':
    # #             learning_rate = 0.0005
    # #             n_layers = 2
    # #             n_heads = 1
    # #             hidden_dropout_prob = 0.25
    # #             attn_dropout_prob = 0.25
    # #         elif config['dataset'] == 'movie_tv':
    # #             learning_rate = 0.001
    # #             n_layers = 2
    # #             n_heads = 2
    # #             hidden_dropout_prob = 0.25
    # #             attn_dropout_prob = 0.5
    # #         elif config['dataset'] == 'movielen':
    # #             learning_rate = 0.0005
    # #             n_layers = 3
    # #             n_heads = 2
    # #             hidden_dropout_prob = 0
    # #             attn_dropout_prob = 0.25
    # #         elif config['dataset'] == 'kindle':
    # #             learning_rate = 0.0005
    # #             n_layers = 3
    # #             n_heads = 2
    # #             hidden_dropout_prob = 0.25
    # #             attn_dropout_prob = 0
    # #         elif config['dataset'] == 'home':
    # #             learning_rate = 0.001
    # #             n_layers = 3
    # #             n_heads = 2
    # #             hidden_dropout_prob = 0.25
    # #             attn_dropout_prob = 0
    # #         elif config['dataset'] == 'phone':
    # #             learning_rate = 0.001
    # #             n_layers = 2
    # #             n_heads = 1
    # #             hidden_dropout_prob = 0.25
    # #             attn_dropout_prob = 0.25
    # #         cur_config['learning_rate'] = learning_rate
    # #         cur_config['n_layers'] = n_layers
    # #         cur_config['n_heads'] = n_heads
    # #         cur_config['hidden_dropout_prob'] = hidden_dropout_prob
    # #         cur_config['attn_dropout_prob'] = attn_dropout_prob
    # #         res.append(cur_config)
    #
    #
    # elif model == 'NextItNet':
    #     for learning_rate in learning_rate_lst:
    #         for block in block_lst:
    #             cur_config = config.copy()
    #             cur_config['learning_rate'] = learning_rate
    #             cur_config['block_num'] = block
    #             res.append(cur_config)
    # elif model == 'GRNN':
    #     for learning_rate in learning_rate_lst:
    #         for dropout_prob in dropout_prob_lst:
    #             cur_config = config.copy()
    #             cur_config['learning_rate'] = learning_rate
    #             cur_config['dropout_prob'] = dropout_prob
    #             cur_config['gnn_hidden_dropout_prob'] = 0
    #             cur_config['gnn_att_dropout_prob'] = 0
    #             cur_config['n_layers'] = 1
    #             cur_config['n_heads'] = 1
    #             res.append(cur_config)


    return res


if __name__ == "__main__":

    model = 'SRGNN'
    dataset = 'home'
    gpu_id = 2
    epochs = 300
    train_batch_size = 512

    config = {'model': model,
              'dataset': dataset,
              'gpu_id': gpu_id,
              'epochs': epochs,
              'max_len':50,
              'train_batch_size':train_batch_size,
              'device': 'cuda:'+str(gpu_id) }


    config_path = './config/'+model+'.yaml'
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

    founded_best_hit_10 = 0
    founded_best_ndcg_10 = 0

    founded_best_test_hit_5, founded_best_test_ndcg_5, founded_best_test_mrr_5, \
    founded_best_test_hit_10, founded_best_test_ndcg_10, founded_best_test_mrr_10, \
    founded_best_test_hit_20, founded_best_test_ndcg_20, founded_best_test_mrr_20, =\
    0,0,0,0,0,0,0,0,0

    # TODO 是不是可以从这里循环调参数
    best_config = config.copy()

    config_lst = load_hyper_param(config.copy(),model,dataset)
    hyper_count = len(config_lst)
    logger.info('[hyper parameter count]: %d' % (hyper_count))
    hyper_number = 0

    best_model_dict = None



    for config in config_lst:

        seed_torch()

        hyper_number += 1
        num_items, train_loader, test_loader, dev_loader = load_data(prepare_data_model,config['max_len'])

        logger.info(' start training, running parameters:')
        logger.info(config)

        if config['model'] == 'GRU4Rec':
            model_obj = GRU4Rec(config, num_items)
        elif config['model'] == 'NARM':
            model_obj = NARM(config, num_items)
        elif config['model'] == 'SASRec':
            model_obj = SASRec(config,num_items)
        elif config['model'] =='STAMP':
            model_obj = STAMP(config, num_items)
        elif config['model'] == 'SRGNN':
            model_obj = SRGNN(config,num_items)
        elif config['model'] == 'GCSAN':
            model_obj = GCSAN(config,num_items)
        elif config['model'] == 'NextItNet':
            model_obj = NextItNet(config, num_items)
        elif config['model'] == 'GRNN':
            model_obj = GRNN(config, num_items)
        # elif config['model'] == 'LESSR':
        #     model = LESSR(config, num_items)
        device = config['device']

        model_obj = model_obj.to(device)

        runner = TrainRunnerNormal(
            model_obj,
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
            best_model_dict = runner.get_best_model()


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
        logger.info('[hyper number]: %d/%d'%(hyper_number,hyper_count))

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

        if config['model'] == 'GRU4Rec':
            best_model = GRU4Rec(best_config, num_items)
        elif config['model'] == 'NARM':
            best_model = NARM(best_config, num_items)
        elif config['model'] == 'SASRec':
            best_model = SASRec(best_config,num_items)
        elif config['model'] =='STAMP':
            best_model = STAMP(best_config, num_items)
        elif config['model'] == 'SRGNN':
            best_model = SRGNN(best_config,num_items)
        elif config['model'] == 'GCSAN':
            best_model = GCSAN(best_config,num_items)
        elif config['model'] == 'NextItNet':
            best_model = NextItNet(best_config, num_items)
        elif config['model'] == 'GRNN':
            best_model = GRNN(best_config, num_items)
        # elif config['model'] == 'LESSR':
        #     model = LESSR(config, num_items)
        device = config['device']

        best_model = best_model.to(device)
        best_model.load_state_dict(loaded_model_dict)

        runner = TrainRunnerNormal(
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
