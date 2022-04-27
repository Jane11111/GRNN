# -*- coding: utf-8 -*-
# @Time    : 2020-12-27 18:56
# @Author  : zxl
# @FileName: predict.py


from utils.trainer import TrainRunner
from model.grnn import GRNN
from model.baseline.gru4rec import GRU4Rec
from model.baseline.narm import NARM
from model.baseline.sasrec import SASRec
from model.baseline.stamp import STAMP
from model.grnn import GRNN_only_graph,GRNN_gru_pro
from model.baseline.srgnn import SRGNN
from model.baseline.gcsan import GCSAN
from prepare_data.dataset import PaddedDataset,PaddedDatasetNormal
from prepare_data.dataset import DatasetLessr
from model.baseline.lessr_fast import LESSR_fast
from prepare_data.preprocess import PrepareData
import time
import logging
from prepare_data.collate import collate_fn,collate_fn_normal,collate_fn_lessr
from utils.trainer import TrainRunner,TrainRunnerNormal,TrainRunnerLessr
from torch.utils.data import DataLoader
from logging import getLogger
import torch as th
import yaml
import copy

def load_data(prepare_data_model,model_name,config):

    if model_name == 'GRNN' or model_name == 'GRNN_gru_pro' or model_name == 'GRNN_only_graph':

        train_sessions, test_sessions, dev_sessions = prepare_data_model.read_train_dev_test()
    else:
        train_sessions, test_sessions, dev_sessions = prepare_data_model.read_train_dev_test_normal()
    if model_name == 'GRNN' or model_name == 'GRNN_gru_pro' or model_name == 'GRNN_only_graph':
        fn = collate_fn
        train_set = PaddedDataset(train_sessions, max_len)
        test_set = PaddedDataset(test_sessions, max_len)
        dev_set = PaddedDataset(dev_sessions, max_len)
    elif model_name == 'LESSR_fast':
        fn = collate_fn_lessr
        train_set = DatasetLessr(train_sessions, max_len)
        test_set = DatasetLessr(test_sessions, max_len)
        dev_set = DatasetLessr(dev_sessions, max_len)
    else:
        fn = collate_fn_normal
        train_set = PaddedDatasetNormal(train_sessions, max_len)
        test_set = PaddedDatasetNormal(test_sessions, max_len)
        dev_set = PaddedDatasetNormal(dev_sessions, max_len)


    prepare_data_model.get_statistics()
    num_items = prepare_data_model.get_item_num()


    train_loader = DataLoader(
        train_set,
        batch_size=config['train_batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=1,
        collate_fn=fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config['train_batch_size'],
        shuffle=False,
        num_workers=1,
        collate_fn=fn,
        pin_memory=True
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=config['train_batch_size'],
        shuffle=False,
        num_workers=1,
        collate_fn=fn,
        pin_memory=True
    )
    print('load finished')
    return num_items, train_loader, test_loader, dev_loader

def get_runner(model_name,best_model,train_loader,test_loader, dev_loader,config,device,logger):
    if model_name == 'GRNN' or model_name == 'GRNN_gru_pro' or model_name == 'GRNN_only_graph':
        runner = TrainRunner(
            best_model,  # 最好的model
            train_loader,
            test_loader,
            dev_loader,
            device=device,
            lr=config['learning_rate'],
            weight_decay=0,
            logger=logger,
        )
    elif model_name == 'LESSR_fast':
        runner = TrainRunnerLessr(
            best_model,
            train_loader,
            test_loader,
            dev_loader,
            device=device,
            lr=config['learning_rate'],
            weight_decay=0,
            logger=logger,
        )
    else:
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
    return runner

def get_logger(config):

    cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_path = './data/log/predict_' + config['model'] + '_' + config['dataset'] + '-' + str(cur_time) + "_log.txt"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = getLogger('test')
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s   %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(log_path)
    logger.info(config)
    return logger

def load_hyper_param(config,data_name , model_name):

    res = []
    config_path = './config/' + model_name + '.yaml'
    with open(config_path, 'r') as f:
        dict = yaml.load(f.read(), Loader=yaml.FullLoader)
    for key in dict:
        config[key] = dict[key]

    """
    t test for ablation
    """

    # if model_name == 'GRU4Rec' and data_name == 'movie_tv':
    #         cur_config = config.copy()
    #         cur_config['learning_rate'] = 0.005
    #         cur_config['dropout_prob'] = 0
    #         res.append(cur_config)
    # elif model_name == 'GRNN_gru_pro' and data_name == 'tmall_buy':
    #     embedding_size = 128
    #     hidden_size = 128
    #     gnn_hidden_dropout_prob = 0
    #     gnn_att_dropout_prob = 0
    #     learning_rate = 0.001
    #     dropout_prob = 0.25
    #     agg_layer = 3
    #     cur_config = config.copy()
    #     cur_config['hidden_size'] = hidden_size
    #     cur_config['embedding_size'] = embedding_size
    #     cur_config['gnn_hidden_dropout_prob'] = gnn_hidden_dropout_prob
    #     cur_config['gnn_att_dropout_prob'] = gnn_att_dropout_prob
    #     cur_config['learning_rate'] = learning_rate
    #     cur_config['dropout_prob'] = dropout_prob
    #     cur_config['agg_layer'] = agg_layer
    #     res.append(cur_config)
    # elif model_name == 'GRNN_only_graph' and data_name == 'tmall_buy':
    #     embedding_size = 128
    #     hidden_size = 128
    #     gnn_hidden_dropout_prob = 0
    #     gnn_att_dropout_prob = 0
    #     learning_rate = 0.001
    #     dropout_prob = 0.25
    #     agg_layer = 1
    #     cur_config = config.copy()
    #     cur_config['hidden_size'] = hidden_size
    #     cur_config['embedding_size'] = embedding_size
    #     cur_config['gnn_hidden_dropout_prob'] = gnn_hidden_dropout_prob
    #     cur_config['gnn_att_dropout_prob'] = gnn_att_dropout_prob
    #     cur_config['learning_rate'] = learning_rate
    #     cur_config['dropout_prob'] = dropout_prob
    #     cur_config['agg_layer'] = agg_layer
    #     res.append(cur_config)
    # elif model_name == 'GRNN_only_graph' and data_name == 'home':
    #     embedding_size = 128
    #     hidden_size = 128
    #     gnn_hidden_dropout_prob = 0
    #     gnn_att_dropout_prob = 0
    #     learning_rate = 0.001
    #     dropout_prob = 0.25
    #     agg_layer = 3
    #     cur_config = config.copy()
    #     cur_config['hidden_size'] = hidden_size
    #     cur_config['embedding_size'] = embedding_size
    #     cur_config['gnn_hidden_dropout_prob'] = gnn_hidden_dropout_prob
    #     cur_config['gnn_att_dropout_prob'] = gnn_att_dropout_prob
    #     cur_config['learning_rate'] = learning_rate
    #     cur_config['dropout_prob'] = dropout_prob
    #     cur_config['agg_layer'] = agg_layer
    #     res.append(cur_config)
    # elif model_name == 'GRNN_gru_pro' and data_name == 'home':
    #     embedding_size = 128
    #     hidden_size = 128
    #     gnn_hidden_dropout_prob = 0
    #     gnn_att_dropout_prob = 0
    #     learning_rate = 0.001
    #     dropout_prob = 0.25
    #     agg_layer = 3
    #     cur_config = config.copy()
    #     cur_config['hidden_size'] = hidden_size
    #     cur_config['embedding_size'] = embedding_size
    #     cur_config['gnn_hidden_dropout_prob'] = gnn_hidden_dropout_prob
    #     cur_config['gnn_att_dropout_prob'] = gnn_att_dropout_prob
    #     cur_config['learning_rate'] = learning_rate
    #     cur_config['dropout_prob'] = dropout_prob
    #     cur_config['agg_layer'] = agg_layer
    #     res.append(cur_config)
    if model_name == 'GRNN':
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

        cur_config = config.copy()
        cur_config['hidden_size'] = hidden_size
        cur_config['embedding_size'] = embedding_size
        cur_config['gnn_hidden_dropout_prob'] = gnn_hidden_dropout_prob
        cur_config['gnn_att_dropout_prob'] = gnn_att_dropout_prob
        cur_config['learning_rate'] = learning_rate
        cur_config['dropout_prob'] = dropout_prob
        cur_config['agg_layer'] = agg_layer
        res.append(cur_config)
    # elif model_name == 'LESSR_fast':
    #     if data_name == 'movielen':
    #         lr = 0.001
    #         step = 4
    #     if data_name == 'home':
    #         lr = 0.001
    #         step = 3
    #     cur_config = config.copy()
    #     cur_config['learning_rate'] = lr
    #     cur_config['step'] = step  # SRGNN, GCSAN, LESSR
    #     res.append(cur_config)
    # elif model_name == 'GRU4Rec':
    #     if data_name == 'tmall_buy':
    #         cur_config = config.copy()
    #         cur_config['learning_rate'] = 0.001
    #         cur_config['dropout_prob'] = 0
    #         res.append(cur_config)
    #     elif data_name == 'phone':
    #         cur_config = config.copy()
    #         cur_config['learning_rate'] = 0.005
    #         cur_config['dropout_prob'] = 0.25
    #         res.append(cur_config)
    #     elif data_name == 'home':
    #         cur_config = config.copy()
    #         cur_config['learning_rate'] = 0.005
    #         cur_config['dropout_prob'] = 0.25
    #         res.append(cur_config)
    #     elif data_name == 'elec':
    #         cur_config = config.copy()
    #         cur_config['learning_rate'] = 0.005
    #         cur_config['dropout_prob'] = 0.25
    #         res.append(cur_config)
    # elif model_name == 'NARM':
    #     if data_name == 'movie_tv':
    #         cur_config = config.copy()
    #         cur_config['learning_rate'] = 0.005
    #         cur_config['dropout_probs'] = [0.25,0]
    #         res.append(cur_config)
    #     if data_name == 'home':
    #         cur_config = config.copy()
    #         cur_config['learning_rate'] = 0.005
    #         cur_config['dropout_probs'] = [0, 0]
    #         res.append(cur_config)
    # elif model_name == 'SASRec':
    #     if data_name == 'kindle':
    #         cur_config = copy.deepcopy(config)
    #         cur_config['learning_rate'] = 0.0005
    #         cur_config['n_layers'] = 3
    #         cur_config['n_heads'] = 1
    #         cur_config['hidden_dropout_prob'] = 0.25
    #         cur_config['attn_dropout_prob'] = 0.25
    #         res.append(cur_config)
    # elif model_name == 'STAMP':
    #     if data_name == 'tmall_buy':
    #         cur_config = config.copy()
    #         cur_config['learning_rate'] = 0.001
    #         res.append(cur_config)


    return res


def get_best_model_path(model , dataset):
    root = '/home/zxl/project/GRNN/data/model/'

    """
    t test for ablation
    """
    if dataset == 'movie_tv' and model == 'GRU4Rec':
        model_save_path = root + 'GRU4Rec_movie_tv_2021-05-16_13-59-14.pkl'
    if dataset == 'tmall_buy' and model == 'GRNN_gru_pro':
        model_save_path = root + 'GRNN_gru_pro_tmall_buy_2021-05-16_13-11-36.pkl'
    if dataset == 'tmall_buy' and model == 'GRNN_only_graph':
        model_save_path = root + 'GRNN_only_graph_tmall_buy_2021-05-20_14-55-19.pkl'

    if dataset == 'home' and model == 'GRNN_only_graph':
        model_save_path = root + 'GRNN_only_graph_home_2021-05-16_13-04-14.pkl'
    if dataset == 'home' and model == 'GRNN_gru_pro':
        model_save_path = root + 'GRNN_gru_pro_home_2021-05-16_13-03-56.pkl'


    if dataset == 'movie_tv':
        if model == 'GRNN':
            model_save_path = root + 'GRNN_movie_tv_2020-12-26_20-18-41.pkl'
        if model == 'NARM':
            model_save_path = root + 'NARM_movie_tv_2020-12-26_17-21-41.pkl'
    elif dataset == 'tmall_buy':
        if model == 'GRNN':
            model_save_path = root + 'GRNN_tmall_buy_2020-12-27_11-05-24.pkl'
        if model == 'GRU4Rec':
            model_save_path = root + 'GRU4Rec_tmall_buy_2020-12-26_16-39-46.pkl'
        if model == 'STAMP':
            model_save_path = root + 'STAMP_tmall_buy_2020-12-31_00-07-28.pkl'
    # elif dataset == 'phone':
    #     if model == 'GRNN':
    #         model_save_path = root + 'GRNN_phone_2020-12-26_18-11-31.pkl'
    #     if model == 'GRU4Rec':
    #         model_save_path = root + 'GRU4Rec_phone_2020-12-26_16-50-38.pkl'
    # elif dataset == 'movielen':
    #     if model == 'GRNN':
    #         # model_save_path = root + 'GRNN_movielen_2020-12-27_11-14-55.pkl'
    #         model_save_path = root + 'GRNN_movielen_2020-12-30_19-36-02.pkl'
    #     if model == 'LESSR_fast':
    #         model_save_path = root  + 'LESSR_fast_movielen_2020-12-28_19-13-25.pkl'
    elif dataset == 'home':
        if model == 'GRNN':
            model_save_path = root + 'GRNN_home_2020-12-27_11-35-15.pkl'
        if model == 'LESSR_fast':
            model_save_path = root + 'LESSR_fast_home_2020-12-28_22-14-52.pkl'
        if model == 'NARM':
            model_save_path = root + 'NARM_home_2020-12-30_21-41-05.pkl'
        if model == 'GRU4Rec':
            model_save_path = root + 'GRU4Rec_home_2020-12-30_22-20-35.pkl'
    # elif dataset == 'kindle':
    #     if model == 'SASRec':
    #         model_save_path = root + 'SASRec_kindle_2020-12-26_18-19-08.pkl'
    #     if model == 'GRNN':
    #         model_save_path = root + 'GRNN_kindle_2020-12-28_11-20-12.pkl'
    elif dataset == 'elec':
        if model == 'GRNN':
            model_save_path = root + 'GRNN_elec_2020-12-29_09-12-41.pkl'
        if model == 'GRU4Rec':
            model_save_path = root + 'GRU4Rec_elec_2020-12-26_17-03-51.pkl'

    return model_save_path

def get_best_model(model_name,data_name,logger,config ):
    prepare_data_model = PrepareData(config, logger)
    num_items, train_loader, test_loader, dev_loader = load_data(prepare_data_model, model_name,config)

    model_save_path = get_best_model_path(model_name, data_name)
    loaded_model_dict = th.load(model_save_path)

    if model_name == 'GRNN':
        model_obj = GRNN(config, num_items)
    elif model_name == 'NARM':
        model_obj = NARM(config,num_items)
    elif model_name == 'GRU4Rec':
        model_obj = GRU4Rec(config,num_items)
    elif model_name == 'SASRec':
        model_obj = SASRec(config,num_items)
    elif model_name == 'LESSR_fast':
        model_obj = LESSR_fast(config,num_items)
    elif model_name == 'STAMP':
        model_obj = STAMP(config,num_items)
    elif model_name == 'GRNN_only_graph':
        model_obj = GRNN_only_graph(config,num_items)
    elif model_name == 'GRNN_gru_pro':
        model_obj = GRNN_gru_pro(config,num_items)

    model_obj = model_obj.to(config['device'])

    best_model = model_obj
    best_model.load_state_dict(loaded_model_dict)

    runner = get_runner(model_name, best_model, train_loader, test_loader, dev_loader,config,config['device'],logger)

    return best_model,runner

def get_overall_config_lst(model, dataset,gpu_id):

    epochs = 300
    train_batch_size = 512
    config = {'model': model,
              'dataset': dataset,
              'gpu_id': gpu_id,
              'epochs': epochs,
              'max_len': 50,
              'train_batch_size': train_batch_size,
              'device': 'cuda:' + str(gpu_id)}

    config = load_hyper_param(config, dataset, model)[0]
    return config

def ttest(data_name):
    pass

def prepare(model_name, data_name,gpu_id):
    config = get_overall_config_lst(model_name, data_name,gpu_id)
    logger = get_logger(config)
    best_model, runner = get_best_model(model_name, data_name, logger, config)
    return config, logger, best_model,runner

if __name__ == "__main__":

    max_len = 50
    gpu_id = 2

    model_name = 'GRNN'
    data_name = 'home'
    result_save_path = './data/best_result/'+model_name+'_'+data_name+'-0516.txt'

    config, logger, best_model, runner = prepare(model_name,data_name,gpu_id)

    test_hit_5, test_ndcg_5, test_mrr_5, \
    test_hit_10, test_ndcg_10, test_mrr_10, \
    test_hit_20, test_ndcg_20, test_mrr_20 = \
        runner.evaluate(best_model, runner.test_loader, runner.device)

    logger.info('[model_name: %s, data_name: %s]'%(model_name, data_name))
    logger.info('<loaded best test> hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                % (test_hit_5, test_ndcg_5, test_mrr_5,
                   test_hit_10, test_ndcg_10, test_mrr_10,
                   test_hit_20, test_ndcg_20, test_mrr_20))

    topk_lst = runner.get_topk(best_model,k=50)
    with open(result_save_path,'w') as w:
        for lst in topk_lst:
            w.write(str(lst))
            w.write('\n')
        w.write('<loaded best test> hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                    'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                    'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                    % (test_hit_5, test_ndcg_5, test_mrr_5,
                       test_hit_10, test_ndcg_10, test_mrr_10,
                       test_hit_20, test_ndcg_20, test_mrr_20))