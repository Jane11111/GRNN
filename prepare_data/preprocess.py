# -*- coding: utf-8 -*-
# @Time    : 2020-11-15 15:26
# @Author  : zxl
# @FileName: preprocess.py

import numpy as np
import pandas as pd
import pickle

class PrepareData():
    def __init__(self,config,logger):
        # root = '/home/zxl/project/MTAM-t2/data/'
        root = '/data/zxl/MTAM-t2/data/'

        self.train_path = root + 'training_testing_data/' + config[
            'dataset'] + '_time_item_based_unidirection/train_data_new.pt'
        self.test_path = root + 'training_testing_data/' + config[
            'dataset'] + '_time_item_based_unidirection/test_data_new.pt'
        self.dev_path = root + 'training_testing_data/' + config[
            'dataset'] + '_time_item_based_unidirection/dev_data_new.pt'

        self.normal_train_path = root + 'training_testing_data/'+config[
            'dataset'] + '_time_item_based_unidirection/train_data_new.pt'
        self.normal_test_path = root + 'training_testing_data/'+config[
            'dataset']+'_time_item_based_unidirection/test_data_new.pt'
        self.normal_dev_path = root + 'training_testing_data/'+config[
            'dataset']+'_time_item_based_unidirection/dev_data_new.pt'

        self.parameter_train_path = root + 'training_testing_data/' + config[
            'dataset'] + '_time_item_based_unidirection/train_data_new100.pt'
        self.parameter_test_path = root + 'training_testing_data/' + config[
            'dataset'] + '_time_item_based_unidirection/test_data_new100.pt'
        self.parameter_dev_path = root + 'training_testing_data/' + config[
            'dataset'] + '_time_item_based_unidirection/dev_data_new100.pt'

        self.origin_path = root + 'orgin_data/'+config['dataset']+'.csv'

        self.config=config
        self.logger = logger



     
    # def construct_graph(self,u_input):
    #     position_count = len(u_input)
    #     u_input = np.array(u_input)
    #     u_A_adj = np.zeros((position_count, position_count))
    #
    #     for i in np.arange(len(u_input) - 1):
    #         u_lst = np.where(u_input == u_input[i])[0]
    #         v_lst = np.where(u_input == u_input[i + 1])[0]
    #         u_A_adj[i][i] = 1
    #         u_A_adj[i+1][i+1] = 1
    #         for u in u_lst:
    #             for v in v_lst:
    #                 u_A_adj[u][v] = 1
    #                 u_A_adj[v][u] = 1
    #
    #     return u_A_adj

    def construct_graph_no_duplication(self, u_input):
        # 自环只加一次

        position_count = len(u_input)
        u_input = np.array(u_input)
        # u_A_out = np.zeros(shape=(position_count, position_count), dtype=np.int)
        u_A_in = np.zeros(shape=(position_count, position_count), dtype=np.int)

        item2singleidx = {}
        item2idx = {}

        for i in range(len(u_input)):
            item2singleidx[u_input[i]] = i
            if u_input[i] not in item2idx:
                item2idx[u_input[i]] = []
            item2idx[u_input[i]].append(i)
        processed_items = {}
        for i in range(len(u_input) - 1, -1, -1):
            if u_input[i] in processed_items:
                continue
            processed_items[u_input[i]] = True
            for u in item2idx[u_input[i]]:
                for j in range(i):
                    v_idx = item2singleidx[u_input[j]]
                    u_A_in[u][v_idx] = 1

        u_A_in = u_A_in.tolist()
        return u_A_in, u_A_in


    def construct_graph_gated(self,seq):
        u_A_in = []
        for i in range(len(seq)):
            u_A_in.append([0 for i in range(len(seq))])
        u_A_in = np.array(u_A_in)
        u_A_out = []
        for i in range(len(seq)):
            u_A_out.append([0 for i in range(len(seq))])
        u_A_out = np.array(u_A_out)

        dic = {}
        for i in range(len(seq)):
            if seq[i] not in dic:
                dic[seq[i]] = []
            dic[seq[i]].append(i)

        for i in range(len(seq)):
            u_lst = dic[seq[i]]
            if i > 0:
                v = dic[seq[i - 1]][0]
                for u in u_lst:
                    u_A_in[u][v] += 1

            if i < len(seq) - 1:
                v = dic[seq[i + 1]][0]
                for u in u_lst:
                    u_A_out[u][v] += 1
        in_sum = u_A_in.sum(axis=1, keepdims=True)
        in_sum[in_sum == 0] = 1
        out_sum = u_A_out.sum(axis=1, keepdims=True)
        out_sum[out_sum == 0] = 1
        u_A_in = u_A_in / in_sum
        u_A_out = u_A_out / out_sum

        return u_A_in, u_A_out

    def construct_graph_wgat(self,seq):
        u_A_in = []
        for i in range(len(seq)):
            u_A_in.append([0 for i in range(len(seq))])
        u_A_in = np.array(u_A_in)
        u_A_out = []
        for i in range(len(seq)):
            u_A_out.append([0 for i in range(len(seq))])
        u_A_out = np.array(u_A_out)

        dic = {}
        for i in range(len(seq)):
            if seq[i] not in dic:
                dic[seq[i]] = []
            dic[seq[i]].append(i)

        for i in range(len(seq)):
            u_lst = dic[seq[i]]
            if i > 0:
                v = dic[seq[i - 1]][0]

                for u in u_lst:
                    u_A_in[u][v] += 1
                    u_A_in[u][u_lst[0]] += 1

            if i < len(seq) - 1:
                v = dic[seq[i + 1]][0]
                for u in u_lst:
                    u_A_out[u][v] += 1
                    u_A_out[u][u_lst[0]] += 1



        return u_A_in, u_A_out

    def construct_graph_grnn(self,u_input):
        # 自环只加一次

        position_count = len(u_input)
        u_input = np.array(u_input)
        # u_A_out = np.zeros(shape=(position_count, position_count), dtype=np.int)
        u_A_in = np.zeros(shape=(position_count, position_count), dtype=np.int)

        if len(u_input) == len(set(u_input)):
            for i in np.arange(len(u_input) - 1):
                # u_A_out[i, i:] = 1
                u_A_in[i, :i] = 1
        else:
            # print(u_input)
            item2idx = {}
            for i in range(len(u_input)):
                item = u_input[i]
                lst = np.where(u_input == item)[0]
                item2idx[item] = lst


            processed = {}
            for i in np.arange(len(u_input)):
                # u_lst = np.where(u_input == u_input[i])[0]
                u_lst = item2idx[u_input[i]]
                for j in np.arange(0, i + 1, 1):
                    tuple = (u_input[i], u_input[j])
                    if tuple in processed:
                        continue
                    processed[tuple] = True
                    # v_lst = np.where(u_input == u_input[j])[0]
                    v_lst = item2idx[u_input[j]]
                    for u in u_lst:
                        for v in v_lst:
                            u_A_in[u, v] = 1  # 每个结点只计算一次
        u_A_in = u_A_in.tolist()
        # u_A_out = u_A_out.tolist()
        return u_A_in,u_A_in


    def construct_graph_stargnn(self,item_seq):
        # Mask matrix, shape of [batch_size, max_session_len]
        # mask = item_seq.gt(0)
        # items, n_node, A, alias_inputs = [], [], [], []
        # # max_n_node = item_seq.size(1)
        # max_n_node = len(item_seq)
        # # item_seq = item_seq.cpu().numpy()
        # item_seq = np.array(item_seq)
        # for u_input in item_seq:

        u_input = item_seq
        node = np.unique(u_input)
        max_n_node = len(node)

        items=node.tolist()
        u_A = np.zeros((max_n_node, max_n_node))

        for i in np.arange(len(u_input) - 1):
            if u_input[i + 1] == 0:
                break

            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] = 1

        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)

        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()

        A = u_A
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]
            # A.append(u_A)
            #
            # alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        # The relative coordinates of the item node, shape of [batch_size, max_session_len]
        # alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        # The connecting matrix, shape of [batch_size, max_session_len, 2 * max_session_len]
        # A = torch.FloatTensor(A).to(self.device)
        # The unique item nodes, shape of [batch_size, max_session_len]
        # items = torch.LongTensor(items).to(self.device)

        return alias_inputs, A, items

    def get_statistics(self):

        df = pd.read_csv(self.origin_path)

        user_set = set(df['user_id'].tolist())
        self.logger.info('the user count is: %d'%(len(user_set)))
        item_set = set(df['item_id'].tolist())
        self.logger.info('the item count is: %d'%(len(item_set)))
        behavior_count = df.shape[0]
        self.logger.info('the behavior count is : %d'%(behavior_count))

        behavior_per_user = df.groupby(by=['user_id'], as_index=False)['item_id'].count()
        behavior_per_user = behavior_per_user['item_id'].mean()
        self.logger.info('the avg behavior of each user count is: %.5f'%(behavior_per_user))

        behavior_per_item = df.groupby(by=['item_id'], as_index=False)['user_id'].count()
        behavior_per_item = behavior_per_item['user_id'].mean()
        self.logger.info('the avg behavior of each item count is: %.5f'%(behavior_per_item))

        self.user_count = len(user_set)
        self.item_count = len(item_set)

    def get_item_num(self):
        return self.item_count

    def get_train_test_statisitics(self):
        train_set = self.load_dataset(self.train_path, 10000000)
        test_set = self.load_dataset(self.test_path,10000000)
        dev_set = self.load_dataset(self.dev_path,10000000)
        print('[Simple] train: %d, test :%d, dev :%d'%(len(train_set),len(test_set),len(dev_set)))

        train_set = self.load_dataset(self.normal_train_path, 10000000)
        test_set = self.load_dataset(self.normal_test_path, 10000000)
        dev_set = self.load_dataset(self.normal_dev_path, 10000000)
        print('[Normal] train: %d, test :%d, dev :%d' % (len(train_set), len(test_set), len(dev_set)))

    def load_dataset_gated(self, path, limit):

        file = open(path, 'rb')
        dataset = pickle.loads(file.read())
        limit = min(limit, len(dataset))
        dataset = dataset[:limit]
        new_dataset = []
        for lst in dataset:
            lst = lst[:-1]
            seq = lst[1]
            u_A_in, u_A_out = self.construct_graph_gated(seq)
            lst.append(u_A_in)
            lst.append(u_A_out)
            new_dataset.append(lst)

        limit = min(limit, len(dataset))
        new_dataset = new_dataset[:limit]
        return new_dataset

    def load_dataset_no_duplication(self, path, limit):

        file = open(path, 'rb')
        dataset = pickle.loads(file.read())
        limit = min(limit, len(dataset))
        dataset = dataset[:limit]
        new_dataset = []
        for lst in dataset:
            lst = lst[:-1]
            seq = lst[1]
            u_A_in, u_A_out = self.construct_graph_no_duplication(seq)
            lst.append(u_A_in)
            lst.append(u_A_out)
            new_dataset.append(lst)

        limit = min(limit, len(dataset))
        new_dataset = new_dataset[:limit]
        return new_dataset


    def load_dataset_wgat(self, path, limit):

        file = open(path, 'rb')
        dataset = pickle.loads(file.read())
        limit = min(limit, len(dataset))
        dataset = dataset[:limit]
        new_dataset = []
        for lst in dataset:
            lst = lst[:-1]
            seq = lst[1]
            u_A_in, u_A_out = self.construct_graph_wgat(seq)
            lst.append(u_A_in)
            lst.append(u_A_out)
            new_dataset.append(lst)
        limit = min(limit, len(dataset))
        new_dataset = new_dataset[:limit]
        return new_dataset


    def load_dataset_stargnn(self, path, limit):

        file = open(path, 'rb')
        dataset = pickle.loads(file.read())
        limit = min(limit, len(dataset))
        dataset = dataset[:limit]
        new_dataset = []
        for lst in dataset:
            lst = lst[:-1]
            seq = lst[1]
            alias_inputs, A, items  = self.construct_graph_stargnn(seq)
            lst.append(items)
            lst.append(alias_inputs)
            lst.append(A)

            # lst.append(mask)
            new_dataset.append(lst)
        limit = min(limit, len(dataset))
        new_dataset = new_dataset[:limit]
        return new_dataset


    def load_dataset_grnn_parameter(self, path, limit,max_len):

        file = open(path, 'rb')
        dataset = pickle.loads(file.read())
        limit = min(limit, len(dataset))
        dataset = dataset[:limit]
        new_dataset = []
        for lst in dataset:


            user_id = lst[0]
            item_lst = lst[1][-max_len:]
            target_id = lst[2]
            length = len(item_lst)

            new_lst = [user_id,item_lst,target_id,length]

            u_A_in, u_A_out = self.construct_graph_grnn(item_lst)
            new_lst.append(u_A_in)
            new_lst.append(u_A_out)
            new_dataset.append(new_lst)
        limit = min(limit, len(dataset))
        new_dataset = new_dataset[:limit]
        return new_dataset

    def load_dataset(self,path, limit):

        file = open(path, 'rb')
        dataset = pickle.loads(file.read())
        limit = min(limit, len(dataset))
        dataset = dataset[:limit]


        return dataset

    def load_dataset_normal(self,path,max_len, limit):

        file = open(path, 'rb')
        dataset = pickle.loads(file.read())

        new_dataset = []
        for lst in dataset:
            user_id = lst[0]
            item_lst = lst[1][-max_len:]
            target_id = lst[2]
            length = len(item_lst)

            new_lst = [user_id, item_lst, target_id, length]


            new_dataset.append(new_lst)

        dataset = new_dataset

        limit = min(limit, len(dataset))
        dataset = dataset[:limit]




        return dataset

    def read_train_dev_test(self):

        train_limit = 10000000
        test_limit = 20000
        dev_limit = 20000
        # train_limit = 200
        # test_limit = 200
        # dev_limit = 200

        self.train_set  = self.load_dataset(self.train_path,train_limit)
        self.test_set = self.load_dataset(self.test_path, test_limit)
        self.dev_set = self.load_dataset(self.dev_path, dev_limit)

        self.logger.info('train length: %d, dev_length: %d, test_length: %d'%(len(self.train_set),len(self.dev_set), len(self.test_set)))


        return self.train_set, self.test_set, self.dev_set
    def read_train_dev_test_normal(self,max_len):

        train_limit = 10000000
        test_limit = 20000
        dev_limit = 20000
        # train_limit = 200
        # test_limit = 200
        # dev_limit = 200

        self.train_set  = self.load_dataset_normal(self.normal_train_path,max_len,train_limit)
        self.test_set = self.load_dataset_normal(self.normal_test_path, max_len,test_limit)
        self.dev_set = self.load_dataset_normal(self.normal_dev_path, max_len,dev_limit)

        self.logger.info('train length: %d, dev_length: %d, test_length: %d'%(len(self.train_set),len(self.dev_set), len(self.test_set)))


        return self.train_set, self.test_set, self.dev_set


    def read_train_dev_test_gated(self):

        train_limit = 10000000
        test_limit = 20000
        dev_limit = 20000
        # train_limit = 2000
        # test_limit = 2000
        # dev_limit = 2000

        self.train_set  = self.load_dataset_gated(self.normal_train_path,train_limit)
        self.test_set = self.load_dataset_gated(self.normal_test_path, test_limit)
        self.dev_set = self.load_dataset_gated(self.normal_dev_path, dev_limit)

        self.logger.info('train length: %d, dev_length: %d, test_length: %d'%(len(self.train_set),len(self.dev_set), len(self.test_set)))


        return self.train_set, self.test_set, self.dev_set

    def read_train_dev_test_wgat(self):

        train_limit = 10000000
        test_limit = 20000
        dev_limit = 20000
        # train_limit = 2000
        # test_limit = 2000
        # dev_limit = 2000

        self.train_set  = self.load_dataset_wgat(self.normal_train_path,train_limit)
        self.test_set = self.load_dataset_wgat(self.normal_test_path, test_limit)
        self.dev_set = self.load_dataset_wgat(self.normal_dev_path, dev_limit)

        self.logger.info('train length: %d, dev_length: %d, test_length: %d'%(len(self.train_set),len(self.dev_set), len(self.test_set)))


        return self.train_set, self.test_set, self.dev_set
    def read_train_dev_test_stargnn(self):

        train_limit = 10000000
        test_limit = 20000
        dev_limit = 20000
        # train_limit = 20000
        # test_limit = 20000
        # dev_limit = 20000

        self.train_set  = self.load_dataset_stargnn(self.normal_train_path,train_limit)
        self.test_set = self.load_dataset_stargnn(self.normal_test_path, test_limit)
        self.dev_set = self.load_dataset_stargnn(self.normal_dev_path, dev_limit)

        self.logger.info('train length: %d, dev_length: %d, test_length: %d'%(len(self.train_set),len(self.dev_set), len(self.test_set)))


        return self.train_set, self.test_set, self.dev_set


    def read_train_dev_test_no_duplication(self):

        train_limit = 10000000
        test_limit = 20000
        dev_limit = 20000
        # train_limit = 2000
        # test_limit = 2000
        # dev_limit = 2000

        self.train_set  = self.load_dataset_no_duplication(self.normal_train_path,train_limit)
        self.test_set = self.load_dataset_no_duplication(self.normal_test_path, test_limit)
        self.dev_set = self.load_dataset_no_duplication(self.normal_dev_path, dev_limit)

        self.logger.info('train length: %d, dev_length: %d, test_length: %d'%(len(self.train_set),len(self.dev_set), len(self.test_set)))


        return self.train_set, self.test_set, self.dev_set
    def read_train_dev_test_grnn_parameter(self,max_len):

        train_limit = 10000000
        test_limit = 20000
        dev_limit = 20000
        # train_limit = 200
        # test_limit = 200
        # dev_limit = 200

        self.train_set  = self.load_dataset_grnn_parameter(self.normal_train_path,train_limit,max_len)
        self.test_set = self.load_dataset_grnn_parameter(self.normal_test_path, test_limit,max_len)
        self.dev_set = self.load_dataset_grnn_parameter(self.normal_dev_path, dev_limit,max_len)

        self.logger.info('train length: %d, dev_length: %d, test_length: %d'%(len(self.train_set),len(self.dev_set), len(self.test_set)))


        return self.train_set, self.test_set, self.dev_set