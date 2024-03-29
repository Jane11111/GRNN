# -*- coding: utf-8 -*-
# @Time    : 2020-11-15 11:58
# @Author  : zxl
# @FileName: trainer.py


import time

import torch as th
from torch import nn, optim
import numpy as np
import copy
# ignore weight decay for bias and batch norm
def fix_weight_decay(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'batch_norm' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params


def prepare_batch(inputs, device):

    inputs_gpu = [x.to(device) for x in inputs]
    return inputs_gpu

class TrainRunnerGnn:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        dev_loader, # TODO
        device,
        logger,
        lr=1e-3,
        weight_decay=0,
        patience=3,
    ):
        self.model = model
        self.best_model = copy.deepcopy(model.state_dict())
        if weight_decay > 0:
            params = fix_weight_decay(model)
            self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dev_loader = dev_loader
        self.device = device
        self.epoch = 0
        self.batch = 0
        self.patience = patience
        self.logger = logger
        self.best_ndcg_10 = 0.
        self.best_hr_10 = 0.
        self.early_stop = 0
        self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5, \
        self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10, \
        self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20,=0,0,0,0,0,0,0,0,0

    def evaluate(self,model, data_loader, device):
        model.eval()

        mrr_5 = th.tensor(0.0).to(device)
        hit_5 = th.tensor(0.0).to(device)
        ndcg_5 = th.tensor(0.0).to(device)
        mrr_10 = th.tensor(0.0).to(device)
        hit_10 = th.tensor(0.0).to(device)
        ndcg_10 = th.tensor(0.0).to(device)
        mrr_20 = th.tensor(0.0).to(device)
        hit_20 = th.tensor(0.0).to(device)
        ndcg_20 = th.tensor(0.0).to(device)
        num_samples = 0
        log2 = th.log(th.tensor(2.)).to(device)
        with th.no_grad():
            for batch in data_loader:
                user_id, item_seq, target_id, item_seq_len, \
                adj_in,adj_out  = prepare_batch(batch, device)
                logits = model.calculate_logits(item_seq, item_seq_len, adj_in,adj_out )
                batch_size = logits.size(0)
                num_samples += batch_size
                labels = target_id.unsqueeze(-1)

                _, topk = logits.topk(k=5)
                hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
                r_ranks = 1 / hit_ranks.to(th.float32)
                hit_5 += hit_ranks.numel()
                mrr_5 += r_ranks.sum()
                ndcg_5 += (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

                _, topk = logits.topk(k=10)
                hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
                r_ranks = 1 / hit_ranks.to(th.float32)
                hit_10 += hit_ranks.numel()
                mrr_10 += r_ranks.sum()
                ndcg_10 += (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

                _, topk = logits.topk(k=20)
                hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
                r_ranks = 1 / hit_ranks.to(th.float32)
                hit_20 += hit_ranks.numel()
                mrr_20 += r_ranks.sum()
                ndcg_20 += (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

        return hit_5 / num_samples, ndcg_5 / num_samples, mrr_5 / num_samples, \
               hit_10 / num_samples, ndcg_10 / num_samples, mrr_10 / num_samples, \
               hit_20 / num_samples, ndcg_20 / num_samples, mrr_20 / num_samples

    def train(self, epochs ):

        # mean_loss = 0
        for epoch in range(epochs):
            # i = 0
            # for name, param in self.model.named_parameters():
            #     i += 1
            #     if i == 100:
            #         print(name, '      ', param)

            self.model.train()
            for batch in self.train_loader:
                user_id, item_seq, target_id, item_seq_len,\
                adj_in,adj_out = prepare_batch(batch, self.device)
                self.optimizer.zero_grad()

                logits = self.model.calculate_logits(item_seq, item_seq_len, adj_in,adj_out )
                loss = self.criterion(logits, target_id)

                loss.backward()

                self.optimizer.step()
                self.batch += 1
            # TODO Early Stop
            # TODO @k

            dev_hit_5, dev_ndcg_5, dev_mrr_5, \
            dev_hit_10, dev_ndcg_10, dev_mrr_10, \
            dev_hit_20, dev_ndcg_20, dev_mrr_20 = self.evaluate(self.model, self.dev_loader, self.device)

            test_hit_5  , test_ndcg_5  , test_mrr_5  , \
            test_hit_10  , test_ndcg_10  , test_mrr_10 , \
            test_hit_20  , test_ndcg_20 , test_mrr_20  = self.evaluate(self.model, self.test_loader, self.device)

            if dev_hit_10 > self.best_hr_10 and dev_ndcg_10 > self.best_ndcg_10:
                self.best_hr_10 = dev_hit_10
                self.best_ndcg_10 = dev_ndcg_10
                self.best_model = copy.deepcopy(self.model.state_dict())
                # print(self.best_model is self.model)

                self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5, \
                self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10, \
                self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20 = \
                    test_hit_5, test_ndcg_5, test_mrr_5, \
                    test_hit_10, test_ndcg_10, test_mrr_10, \
                    test_hit_20, test_ndcg_20, test_mrr_20

                self.early_stop = 0
            else:
                self.early_stop += 1

            self.logger.info('training printing epoch: %d' % epoch)
            self.logger.info('[dev] hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                             'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                             'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f' % (dev_hit_5, dev_ndcg_5, dev_mrr_5,
                                                                            dev_hit_10, dev_ndcg_10, dev_mrr_10,
                                                                            dev_hit_20, dev_ndcg_20, dev_mrr_20))
            self.logger.info('[test] hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                             'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                             'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                             % (test_hit_5, test_ndcg_5, test_mrr_5,
                                test_hit_10, test_ndcg_10, test_mrr_10,
                                test_hit_20, test_ndcg_20, test_mrr_20))

            self.logger.info('[best test] hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                             'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                             'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                             % (self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5,
                                self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10,
                                self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20))

            if self.early_stop >= self.patience:
                break
        return self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5, \
               self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10, \
               self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20, \
               self.best_hr_10, self.best_ndcg_10
        # TODO 可以返回最好的valid score

    def get_best_model(self):
        return self.best_model
    def get_top1(self,model,  ):
        model.eval()
        device = self.device
        data_loader = self.test_loader

        top1_lst = []
        with th.no_grad():
            for batch in data_loader:
                user_id, item_seq, target_id, item_seq_len, \
                adj_in, adj_out  = prepare_batch(batch, device)
                logits = model.calculate_logits(item_seq, item_seq_len, adj_in,adj_out )

                _, topk = logits.topk(k=1)
                top1_lst.extend(list(topk.cpu().numpy().reshape(-1)))
        return top1_lst
    def get_topk(self,model,  k):
        model.eval()
        device = self.device
        data_loader = self.test_loader

        topk_lst = []
        with th.no_grad():
            for batch in data_loader:
                user_id, item_seq, target_id, item_seq_len, \
                adj_in,adj_out  = prepare_batch(batch, device)
                logits = model.calculate_logits(item_seq, item_seq_len, adj_in,adj_out )

                _, topk = logits.topk(k=k)
                topk_lst.extend( topk.cpu().numpy().tolist())
        return topk_lst

class TrainRunnerStarGnn:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        dev_loader, # TODO
        device,
        logger,
        lr=1e-3,
        weight_decay=0,
        patience=3,
    ):
        self.model = model
        self.best_model = copy.deepcopy(model.state_dict())
        if weight_decay > 0:
            params = fix_weight_decay(model)
            self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dev_loader = dev_loader
        self.device = device
        self.epoch = 0
        self.batch = 0
        self.patience = patience
        self.logger = logger
        self.best_ndcg_10 = 0.
        self.best_hr_10 = 0.
        self.early_stop = 0
        self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5, \
        self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10, \
        self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20,=0,0,0,0,0,0,0,0,0

    def evaluate(self,model, data_loader, device):
        model.eval()

        mrr_5 = th.tensor(0.0).to(device)
        hit_5 = th.tensor(0.0).to(device)
        ndcg_5 = th.tensor(0.0).to(device)
        mrr_10 = th.tensor(0.0).to(device)
        hit_10 = th.tensor(0.0).to(device)
        ndcg_10 = th.tensor(0.0).to(device)
        mrr_20 = th.tensor(0.0).to(device)
        hit_20 = th.tensor(0.0).to(device)
        ndcg_20 = th.tensor(0.0).to(device)
        num_samples = 0
        log2 = th.log(th.tensor(2.)).to(device)
        with th.no_grad():
            for batch in data_loader:
                user_id, item_seq, target_id, item_seq_len, \
                items,alias_inputs,A  = prepare_batch(batch, device)

                logits = model.calculate_logits(items,alias_inputs,A, item_seq_len)
                batch_size = logits.size(0)
                num_samples += batch_size
                labels = target_id.unsqueeze(-1)

                _, topk = logits.topk(k=5)
                hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
                r_ranks = 1 / hit_ranks.to(th.float32)
                hit_5 += hit_ranks.numel()
                mrr_5 += r_ranks.sum()
                ndcg_5 += (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

                _, topk = logits.topk(k=10)
                hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
                r_ranks = 1 / hit_ranks.to(th.float32)
                hit_10 += hit_ranks.numel()
                mrr_10 += r_ranks.sum()
                ndcg_10 += (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

                _, topk = logits.topk(k=20)
                hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
                r_ranks = 1 / hit_ranks.to(th.float32)
                hit_20 += hit_ranks.numel()
                mrr_20 += r_ranks.sum()
                ndcg_20 += (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

        return hit_5 / num_samples, ndcg_5 / num_samples, mrr_5 / num_samples, \
               hit_10 / num_samples, ndcg_10 / num_samples, mrr_10 / num_samples, \
               hit_20 / num_samples, ndcg_20 / num_samples, mrr_20 / num_samples

    def train(self, epochs ):

        # mean_loss = 0
        for epoch in range(epochs):
            # i = 0
            # for name, param in self.model.named_parameters():
            #     i += 1
            #     if i == 100:
            #         print(name, '      ', param)

            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                user_id, item_seq, target_id, item_seq_len, \
                items, alias_inputs, A = prepare_batch(batch, self.device)
                self.optimizer.zero_grad()

                logits = self.model.calculate_logits(items,alias_inputs,A, item_seq_len )
                loss = self.criterion(logits, target_id)
                # print(float(loss.data))
                total_loss += float(loss.data)

                loss.backward()

                self.optimizer.step()
                self.batch += 1
            # TODO Early Stop
            # TODO @k
            # print(total_loss)

            dev_hit_5, dev_ndcg_5, dev_mrr_5, \
            dev_hit_10, dev_ndcg_10, dev_mrr_10, \
            dev_hit_20, dev_ndcg_20, dev_mrr_20 = self.evaluate(self.model, self.dev_loader, self.device)

            test_hit_5  , test_ndcg_5  , test_mrr_5  , \
            test_hit_10  , test_ndcg_10  , test_mrr_10 , \
            test_hit_20  , test_ndcg_20 , test_mrr_20  = self.evaluate(self.model, self.test_loader, self.device)

            if dev_hit_10 > self.best_hr_10 and dev_ndcg_10 > self.best_ndcg_10:
                self.best_hr_10 = dev_hit_10
                self.best_ndcg_10 = dev_ndcg_10
                self.best_model = copy.deepcopy(self.model.state_dict())
                # print(self.best_model is self.model)

                self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5, \
                self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10, \
                self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20 = \
                    test_hit_5, test_ndcg_5, test_mrr_5, \
                    test_hit_10, test_ndcg_10, test_mrr_10, \
                    test_hit_20, test_ndcg_20, test_mrr_20

                self.early_stop = 0
            else:
                self.early_stop += 1

            self.logger.info('training printing epoch: %d' % epoch)
            self.logger.info('[dev] hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                             'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                             'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f' % (dev_hit_5, dev_ndcg_5, dev_mrr_5,
                                                                            dev_hit_10, dev_ndcg_10, dev_mrr_10,
                                                                            dev_hit_20, dev_ndcg_20, dev_mrr_20))
            self.logger.info('[test] hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                             'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                             'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                             % (test_hit_5, test_ndcg_5, test_mrr_5,
                                test_hit_10, test_ndcg_10, test_mrr_10,
                                test_hit_20, test_ndcg_20, test_mrr_20))

            self.logger.info('[best test] hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                             'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                             'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                             % (self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5,
                                self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10,
                                self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20))

            if self.early_stop >= self.patience:
                break
        return self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5, \
               self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10, \
               self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20, \
               self.best_hr_10, self.best_ndcg_10
        # TODO 可以返回最好的valid score

    def get_best_model(self):
        return self.best_model
    def get_top1(self,model,  ):
        model.eval()
        device = self.device
        data_loader = self.test_loader

        top1_lst = []
        with th.no_grad():
            for batch in data_loader:
                user_id, item_seq, target_id, item_seq_len, \
                items, alias_inputs, A  = prepare_batch(batch, device)
                logits = model.calculate_logits(items,alias_inputs,A, item_seq_len )

                _, topk = logits.topk(k=1)
                top1_lst.extend(list(topk.cpu().numpy().reshape(-1)))
        return top1_lst
    def get_topk(self,model,  k):
        model.eval()
        device = self.device
        data_loader = self.test_loader

        topk_lst = []
        with th.no_grad():
            for batch in data_loader:
                user_id, item_seq, target_id, item_seq_len, \
                items, alias_inputs, A  = prepare_batch(batch, device)
                logits = model.calculate_logits(items,alias_inputs,A, item_seq_len)

                _, topk = logits.topk(k=k)
                topk_lst.extend( topk.cpu().numpy().tolist())
        return topk_lst


class TrainRunner:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        dev_loader, # TODO
        device,
        logger,
        lr=1e-3,
        weight_decay=0,
        patience=3,
    ):
        self.model = model
        self.best_model = copy.deepcopy(model.state_dict())
        if weight_decay > 0:
            params = fix_weight_decay(model)
            self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dev_loader = dev_loader
        self.device = device
        self.epoch = 0
        self.batch = 0
        self.patience = patience
        self.logger = logger
        self.best_ndcg_10 = 0.
        self.best_hr_10 = 0.
        self.early_stop = 0
        self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5, \
        self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10, \
        self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20,=0,0,0,0,0,0,0,0,0

    def evaluate(self,model, data_loader, device):
        model.eval()

        mrr_5 = th.tensor(0.0).to(device)
        hit_5 = th.tensor(0.0).to(device)
        ndcg_5 = th.tensor(0.0).to(device)
        mrr_10 = th.tensor(0.0).to(device)
        hit_10 = th.tensor(0.0).to(device)
        ndcg_10 = th.tensor(0.0).to(device)
        mrr_20 = th.tensor(0.0).to(device)
        hit_20 = th.tensor(0.0).to(device)
        ndcg_20 = th.tensor(0.0).to(device)
        num_samples = 0
        log2 = th.log(th.tensor(2.)).to(device)
        with th.no_grad():
            for batch in data_loader:
                user_id, item_seq, target_id, item_seq_len, \
                adj_in  = prepare_batch(batch, device)
                logits = model.calculate_logits(item_seq, item_seq_len, adj_in )
                batch_size = logits.size(0)
                num_samples += batch_size
                labels = target_id.unsqueeze(-1)

                _, topk = logits.topk(k=5)
                hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
                r_ranks = 1 / hit_ranks.to(th.float32)
                hit_5 += hit_ranks.numel()
                mrr_5 += r_ranks.sum()
                ndcg_5 += (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

                _, topk = logits.topk(k=10)
                hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
                r_ranks = 1 / hit_ranks.to(th.float32)
                hit_10 += hit_ranks.numel()
                mrr_10 += r_ranks.sum()
                ndcg_10 += (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

                _, topk = logits.topk(k=20)
                hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
                r_ranks = 1 / hit_ranks.to(th.float32)
                hit_20 += hit_ranks.numel()
                mrr_20 += r_ranks.sum()
                ndcg_20 += (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

        return hit_5 / num_samples, ndcg_5 / num_samples, mrr_5 / num_samples, \
               hit_10 / num_samples, ndcg_10 / num_samples, mrr_10 / num_samples, \
               hit_20 / num_samples, ndcg_20 / num_samples, mrr_20 / num_samples

    def train(self, epochs ):

        # mean_loss = 0
        for epoch in range(epochs):
            # i = 0
            # for name, param in self.model.named_parameters():
            #     i += 1
            #     if i == 100:
            #         print(name, '      ', param)

            self.model.train()
            for batch in self.train_loader:
                user_id, item_seq, target_id, item_seq_len,\
                adj_in = prepare_batch(batch, self.device)
                self.optimizer.zero_grad()

                logits = self.model.calculate_logits(item_seq, item_seq_len, adj_in )
                loss = self.criterion(logits, target_id)

                loss.backward()

                self.optimizer.step()
                self.batch += 1
            # TODO Early Stop
            # TODO @k

            dev_hit_5, dev_ndcg_5, dev_mrr_5, \
            dev_hit_10, dev_ndcg_10, dev_mrr_10, \
            dev_hit_20, dev_ndcg_20, dev_mrr_20 = self.evaluate(self.model, self.dev_loader, self.device)

            test_hit_5  , test_ndcg_5  , test_mrr_5  , \
            test_hit_10  , test_ndcg_10  , test_mrr_10 , \
            test_hit_20  , test_ndcg_20 , test_mrr_20  = self.evaluate(self.model, self.test_loader, self.device)

            if dev_hit_10 > self.best_hr_10 and dev_ndcg_10 > self.best_ndcg_10:
                self.best_hr_10 = dev_hit_10
                self.best_ndcg_10 = dev_ndcg_10
                self.best_model = copy.deepcopy(self.model.state_dict())
                # print(self.best_model is self.model)

                self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5, \
                self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10, \
                self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20 = \
                    test_hit_5, test_ndcg_5, test_mrr_5, \
                    test_hit_10, test_ndcg_10, test_mrr_10, \
                    test_hit_20, test_ndcg_20, test_mrr_20

                self.early_stop = 0
            else:
                self.early_stop += 1

            self.logger.info('training printing epoch: %d' % epoch)
            self.logger.info('[dev] hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                             'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                             'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f' % (dev_hit_5, dev_ndcg_5, dev_mrr_5,
                                                                            dev_hit_10, dev_ndcg_10, dev_mrr_10,
                                                                            dev_hit_20, dev_ndcg_20, dev_mrr_20))
            self.logger.info('[test] hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                             'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                             'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                             % (test_hit_5, test_ndcg_5, test_mrr_5,
                                test_hit_10, test_ndcg_10, test_mrr_10,
                                test_hit_20, test_ndcg_20, test_mrr_20))

            self.logger.info('[best test] hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                             'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                             'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                             % (self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5,
                                self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10,
                                self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20))

            if self.early_stop >= self.patience:
                break
        return self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5, \
               self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10, \
               self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20, \
               self.best_hr_10, self.best_ndcg_10
        # TODO 可以返回最好的valid score

    def get_best_model(self):
        return self.best_model
    def get_top1(self,model,  ):
        model.eval()
        device = self.device
        data_loader = self.test_loader

        top1_lst = []
        with th.no_grad():
            for batch in data_loader:
                user_id, item_seq, target_id, item_seq_len, \
                adj_in  = prepare_batch(batch, device)
                logits = model.calculate_logits(item_seq, item_seq_len, adj_in )

                _, topk = logits.topk(k=1)
                top1_lst.extend(list(topk.cpu().numpy().reshape(-1)))
        return top1_lst
    def get_topk(self,model,  k):
        model.eval()
        device = self.device
        data_loader = self.test_loader

        topk_lst = []
        with th.no_grad():
            for batch in data_loader:
                user_id, item_seq, target_id, item_seq_len, \
                adj_in  = prepare_batch(batch, device)
                logits = model.calculate_logits(item_seq, item_seq_len, adj_in )

                _, topk = logits.topk(k=k)
                topk_lst.extend( topk.cpu().numpy().tolist())
        return topk_lst

class TrainRunnerNormal:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        dev_loader, # TODO
        device,
        logger,
        lr=1e-3,
        weight_decay=0,
        patience=3,
    ):
        self.model = model
        self.best_model = copy.deepcopy(model.state_dict())

        if weight_decay > 0:
            params = fix_weight_decay(model)
            self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dev_loader = dev_loader
        self.device = device
        self.epoch = 0
        self.batch = 0
        self.patience = patience
        self.logger = logger
        self.best_ndcg_10 = 0.
        self.best_hr_10 = 0.
        self.early_stop = 0
        self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5, \
        self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10, \
        self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20,=0,0,0,0,0,0,0,0,0

    def evaluate(self,model, data_loader, device):
        model.eval()

        mrr_5 = th.tensor(0.0).to(device)
        hit_5 = th.tensor(0.0).to(device)
        ndcg_5 = th.tensor(0.0).to(device)
        mrr_10 = th.tensor(0.0).to(device)
        hit_10 = th.tensor(0.0).to(device)
        ndcg_10 = th.tensor(0.0).to(device)
        mrr_20 = th.tensor(0.0).to(device)
        hit_20 = th.tensor(0.0).to(device)
        ndcg_20 = th.tensor(0.0).to(device)
        num_samples = 0
        log2 = th.log(th.tensor(2.)).to(device)
        with th.no_grad():
            for batch in data_loader:
                user_id, item_seq, target_id, item_seq_len = prepare_batch(batch, device)
                logits = model.calculate_logits(item_seq, item_seq_len )
                batch_size = logits.size(0)
                num_samples += batch_size
                labels = target_id.unsqueeze(-1)

                _, topk = logits.topk(k=5)
                hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
                r_ranks = 1 / hit_ranks.to(th.float32)
                hit_5 += hit_ranks.numel()
                mrr_5 += r_ranks.sum()
                ndcg_5 += (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

                _, topk = logits.topk(k=10)
                hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
                r_ranks = 1 / hit_ranks.to(th.float32)
                hit_10 += hit_ranks.numel()
                mrr_10 += r_ranks.sum()
                ndcg_10 += (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

                _, topk = logits.topk(k=20)
                hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
                r_ranks = 1 / hit_ranks.to(th.float32)
                hit_20 += hit_ranks.numel()
                mrr_20 += r_ranks.sum()
                ndcg_20 += (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

        return hit_5 / num_samples, ndcg_5 / num_samples, mrr_5 / num_samples, \
               hit_10 / num_samples, ndcg_10 / num_samples, mrr_10 / num_samples, \
               hit_20 / num_samples, ndcg_20 / num_samples, mrr_20 / num_samples

    def train(self, epochs ):



        # mean_loss = 0
        for epoch in range(epochs):

            # i = 0
            # for name, param in self.model.named_parameters():
            #     i += 1
            #     if i == 5:
            #         print(name, '      ', param)
            # print(i)
            # print('***************')
            self.model.train()
            for batch in self.train_loader:
                user_id, item_seq, target_id, item_seq_len = prepare_batch(batch, self.device)
                self.optimizer.zero_grad()
                logits = self.model.calculate_logits(item_seq, item_seq_len )
                loss = self.criterion(logits, target_id)
                loss.backward()
                self.optimizer.step()

                self.batch += 1
            # TODO Early Stop
            # TODO @k

            dev_hit_5, dev_ndcg_5, dev_mrr_5, \
            dev_hit_10, dev_ndcg_10, dev_mrr_10, \
            dev_hit_20, dev_ndcg_20, dev_mrr_20 = self.evaluate(self.model, self.dev_loader, self.device)

            test_hit_5  , test_ndcg_5  , test_mrr_5  , \
            test_hit_10  , test_ndcg_10  , test_mrr_10 , \
            test_hit_20  , test_ndcg_20 , test_mrr_20  = self.evaluate(self.model, self.test_loader, self.device)



            if dev_hit_10 > self.best_hr_10 and dev_ndcg_10 > self.best_ndcg_10:
                self.best_hr_10 = dev_hit_10
                self.best_ndcg_10 = dev_ndcg_10
                self.best_model = copy.deepcopy(self.model.state_dict())


                self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5, \
                self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10, \
                self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20 = \
                    test_hit_5, test_ndcg_5, test_mrr_5, \
                    test_hit_10, test_ndcg_10, test_mrr_10, \
                    test_hit_20, test_ndcg_20, test_mrr_20

                self.early_stop = 0
            else:
                self.early_stop +=1

            self.logger.info('training printing epoch: %d'%epoch)
            self.logger.info('[dev] hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                             'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                             'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'%( dev_hit_5, dev_ndcg_5, dev_mrr_5,
                                                                          dev_hit_10, dev_ndcg_10, dev_mrr_10,
                                                                          dev_hit_20, dev_ndcg_20, dev_mrr_20))
            self.logger.info('[test] hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                             'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                             'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                             % ( test_hit_5  , test_ndcg_5, test_mrr_5,
                                test_hit_10  , test_ndcg_10  , test_mrr_10 ,
                                test_hit_20  , test_ndcg_20 , test_mrr_20 ))

            self.logger.info('[best test] hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                             'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                             'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                             % ( self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5,
                                self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10,
                                self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20))

            if self.early_stop>=self.patience:
                break
        return self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5, \
        self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10, \
        self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20,\
        self.best_hr_10, self.best_ndcg_10
        # TODO 可以返回最好的valid score
    def get_top1(self,model,  ):
        model.eval()
        device = self.device
        data_loader = self.test_loader

        top1_lst = []
        with th.no_grad():
            for batch in data_loader:
                user_id, item_seq, target_id, item_seq_len = prepare_batch(batch, device)
                logits = model.calculate_logits(item_seq, item_seq_len )

                _, topk = logits.topk(k=1)
                top1_lst.extend(list(topk.cpu().numpy().reshape(-1)))
        return top1_lst
    def get_best_model(self):
        return self.best_model

    def get_topk(self,model,  k):
        model.eval()
        device = self.device
        data_loader = self.test_loader

        topk_lst = []
        with th.no_grad():
            for batch in data_loader:
                user_id, item_seq, target_id, item_seq_len = prepare_batch(batch, device)
                logits = model.calculate_logits(item_seq, item_seq_len  )

                _, topk = logits.topk(k=k)
                topk_lst.extend( topk.cpu().numpy().tolist())
        return topk_lst

class TrainRunnerLessr:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        dev_loader, # TODO
        device,
        logger,
        lr=1e-3,
        weight_decay=0,
        patience=3,
    ):
        self.model = model
        self.best_model = copy.deepcopy(model.state_dict())

        if weight_decay > 0:
            params = fix_weight_decay(model)
            self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dev_loader = dev_loader
        self.device = device
        self.epoch = 0
        self.batch = 0
        self.patience = patience
        self.logger = logger
        self.best_ndcg_10 = 0.
        self.best_hr_10 = 0.
        self.early_stop = 0
        self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5, \
        self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10, \
        self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20,=0,0,0,0,0,0,0,0,0

    def evaluate(self,model, data_loader, device):
        model.eval()

        mrr_5 = th.tensor(0.0).to(device)
        hit_5 = th.tensor(0.0).to(device)
        ndcg_5 = th.tensor(0.0).to(device)
        mrr_10 = th.tensor(0.0).to(device)
        hit_10 = th.tensor(0.0).to(device)
        ndcg_10 = th.tensor(0.0).to(device)
        mrr_20 = th.tensor(0.0).to(device)
        hit_20 = th.tensor(0.0).to(device)
        ndcg_20 = th.tensor(0.0).to(device)
        num_samples = 0
        log2 = th.log(th.tensor(2.)).to(device)
        with th.no_grad():
            for batch in data_loader:
                mg,sg,target_id = prepare_batch(batch, device)
                logits = model.calculate_logits(mg,sg)
                batch_size = logits.size(0)
                num_samples += batch_size
                labels = target_id.unsqueeze(-1)

                _, topk = logits.topk(k=5)
                hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
                r_ranks = 1 / hit_ranks.to(th.float32)
                hit_5 += hit_ranks.numel()
                mrr_5 += r_ranks.sum()
                ndcg_5 += (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

                _, topk = logits.topk(k=10)
                hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
                r_ranks = 1 / hit_ranks.to(th.float32)
                hit_10 += hit_ranks.numel()
                mrr_10 += r_ranks.sum()
                ndcg_10 += (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

                _, topk = logits.topk(k=20)
                hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
                r_ranks = 1 / hit_ranks.to(th.float32)
                hit_20 += hit_ranks.numel()
                mrr_20 += r_ranks.sum()
                ndcg_20 += (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

        return hit_5 / num_samples, ndcg_5 / num_samples, mrr_5 / num_samples, \
               hit_10 / num_samples, ndcg_10 / num_samples, mrr_10 / num_samples, \
               hit_20 / num_samples, ndcg_20 / num_samples, mrr_20 / num_samples

    def train(self, epochs ):

        # mean_loss = 0
        for epoch in range(epochs):
            self.model.train()
            for batch in self.train_loader:
                mg,sg,target_id = prepare_batch(batch, self.device)
                self.optimizer.zero_grad()
                logits = self.model.calculate_logits(mg,sg)
                loss = self.criterion(logits, target_id)
                loss.backward()
                self.optimizer.step()

                self.batch += 1
            # TODO Early Stop
            # TODO @k

            dev_hit_5, dev_ndcg_5, dev_mrr_5, \
            dev_hit_10, dev_ndcg_10, dev_mrr_10, \
            dev_hit_20, dev_ndcg_20, dev_mrr_20 = self.evaluate(self.model, self.dev_loader, self.device)

            test_hit_5  , test_ndcg_5  , test_mrr_5  , \
            test_hit_10  , test_ndcg_10  , test_mrr_10 , \
            test_hit_20  , test_ndcg_20 , test_mrr_20  = self.evaluate(self.model, self.test_loader, self.device)

            if dev_hit_10 > self.best_hr_10 and dev_ndcg_10 > self.best_ndcg_10:
                self.best_hr_10 = dev_hit_10
                self.best_ndcg_10 = dev_ndcg_10
                self.best_model = copy.deepcopy(self.model.state_dict())


                self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5, \
                self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10, \
                self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20 = \
                    test_hit_5, test_ndcg_5, test_mrr_5, \
                    test_hit_10, test_ndcg_10, test_mrr_10, \
                    test_hit_20, test_ndcg_20, test_mrr_20

                self.early_stop = 0
            else:
                self.early_stop += 1

            self.logger.info('training printing epoch: %d' % epoch)
            self.logger.info('[dev] hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                             'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                             'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f' % (dev_hit_5, dev_ndcg_5, dev_mrr_5,
                                                                            dev_hit_10, dev_ndcg_10, dev_mrr_10,
                                                                            dev_hit_20, dev_ndcg_20, dev_mrr_20))
            self.logger.info('[test] hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                             'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                             'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                             % (test_hit_5, test_ndcg_5, test_mrr_5,
                                test_hit_10, test_ndcg_10, test_mrr_10,
                                test_hit_20, test_ndcg_20, test_mrr_20))

            self.logger.info('[best test] hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                             'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                             'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                             % (self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5,
                                self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10,
                                self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20))

            if self.early_stop >= self.patience:
                break
        return self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5, \
               self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10, \
               self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20, \
               self.best_hr_10, self.best_ndcg_10
        # TODO 可以返回最好的valid score
    def get_best_model(self):
        return self.best_model
    def get_top1(self,model,  ):
        model.eval()
        device = self.device
        data_loader = self.test_loader

        top1_lst = []
        with th.no_grad():
            for batch in data_loader:
                mg,sg,target_id = prepare_batch(batch, device)
                logits = model.calculate_logits(mg,sg)
                batch_size = logits.size(0)

                _, topk = logits.topk(k=1)
                top1_lst.extend(list(topk.cpu().numpy().reshape(-1)))
        return top1_lst
    def get_topk(self,model,  k):
        model.eval()
        device = self.device
        data_loader = self.test_loader

        topk_lst = []
        with th.no_grad():
            for batch in data_loader:
                mg, sg, target_id = prepare_batch(batch, device)

                logits = model.calculate_logits(mg, sg)

                _, topk = logits.topk(k=k)
                topk_lst.extend( topk.cpu().numpy().tolist())
        return topk_lst