# -*- coding: utf-8 -*-
# @Time    : 2020-11-15 15:26
# @Author  : zxl
# @FileName: preprocess.py

import numpy as np
import pandas as pd


class PrepareData():
    def __init__(self,config,logger):
        root = '/home/zxl/project/MTAM-t2/data/'

        self.train_path = root + 'training_testing_data/' + config[
            'dataset'] + '_time_item_based_unidirection/train_data_new.txt'
        self.test_path = root + 'training_testing_data/' + config[
            'dataset'] + '_time_item_based_unidirection/test_data_new.txt'
        self.dev_path = root + 'training_testing_data/' + config[
            'dataset'] + '_time_item_based_unidirection/dev_data_new.txt'

        self.normal_train_path = root + 'training_testing_data/'+config[
            'dataset'] + '_time_item_based_unidirection/train_data.txt'
        self.normal_test_path = root + 'training_testing_data/'+config[
            'dataset']+'_time_item_based_unidirection/test_data.txt'
        self.normal_dev_path = root + 'training_testing_data/'+config[
            'dataset']+'_time_item_based_unidirection/dev_data.txt'


        self.origin_path = root + 'orgin_data/'+config['dataset']+'.csv'

        self.config=config
        self.logger = logger



    # def construct_graph(self,u_input):
    #     position_count = len(u_input)
    #     u_input = np.array(u_input)
    #     u_A_out = np.zeros((position_count, position_count))
    #     u_A_in = np.zeros((position_count, position_count))
    #
    #     for i in np.arange(len(u_input) - 2):
    #         u_lst = np.where(u_input == u_input[i])[0]
    #         v_lst = np.where(u_input == u_input[i + 1])[0]
    #         # u_A_out[i][i] += 1
    #         # u_A_in[i][i] += 1
    #
    #
    #         for u in u_lst:
    #             u_A_out[u][v_lst[0]] += 1  # 每个结点只计算一次
    #         for v in v_lst:
    #             u_A_in[v ][u_lst[0]] += 1
    #         # for u in u_lst:  # 自环
    #         #     u_A_out[u][u_lst[0]] += 1
    #         #     u_A_in[u][u_lst[0]] += 1
    #
    #
    #     # print(u_A_out)
    #     u_sum_in = np.reshape(np.sum(u_A_in, 1), (-1, 1))
    #     u_sum_in[np.where(u_sum_in == 0)] = 1
    #     u_A_in = np.divide(u_A_in, u_sum_in)
    #
    #     u_sum_out = np.reshape(np.sum(u_A_out, 1), (-1, 1))
    #     u_sum_out[np.where(u_sum_out == 0)] = 1
    #     u_A_out = np.divide(u_A_out, u_sum_out)
    #
    #     return u_A_in, u_A_out
    def construct_graph(self,u_input):
        position_count = len(u_input)
        u_input = np.array(u_input)
        u_A_adj = np.zeros((position_count, position_count))

        for i in np.arange(len(u_input) - 1):
            u_lst = np.where(u_input == u_input[i])[0]
            v_lst = np.where(u_input == u_input[i + 1])[0]
            u_A_adj[i][i] = 1
            u_A_adj[i+1][i+1] = 1
            for u in u_lst:
                for v in v_lst:
                    u_A_adj[u][v] = 1
                    u_A_adj[v][u] = 1

        return u_A_adj
    # def construct_graph(self, u_input,k=3):
    #
    #     position_count = len(u_input)
    #     u_input = np.array(u_input)
    #     u_A_adj = np.zeros((position_count, position_count))
    #
    #     for i in np.arange(len(u_input) ):
    #         for j in np.arange(max(i-k,0),min(len(u_input),i+k+1),1):
    #             u_A_adj [i][j] = 1
    #
    #     return u_A_adj

    #
    # def construct_graph (self,u_input):
    #     position_count = len(u_input)
    #     u_input = np.array(u_input)
    #     u_A_out = np.zeros((position_count, position_count))
    #     u_A_in = np.zeros((position_count, position_count))
    #
    #     for i in np.arange(len(u_input) - 2):
    #         u_lst = np.where(u_input == u_input[i])[0]
    #         v_lst = np.where(u_input == u_input[i + 1])[0]
    #
    #
    #         for u in u_lst:
    #             u_A_out[u][v_lst[-1]] += 1  # 每个结点只计算一次
    #         for v in v_lst:
    #             u_A_in[v ][u_lst[-1]] += 1
    #
    #
    #
    #     # print(u_A_out)
    #     u_sum_in = np.reshape(np.sum(u_A_in, 1), (-1, 1))
    #     u_sum_in[np.where(u_sum_in == 0)] = 1
    #     u_A_in = np.divide(u_A_in, u_sum_in)
    #
    #     u_sum_out = np.reshape(np.sum(u_A_out, 1), (-1, 1))
    #     u_sum_out[np.where(u_sum_out == 0)] = 1
    #     u_A_out = np.divide(u_A_out, u_sum_out)
    #
    #     return u_A_in, u_A_out

    # def construct_graph(self,u_input):
    #     # 自环只加一次
    #
    #
    #
    #     position_count = len(u_input)
    #     u_input = np.array(u_input)
    #     u_A_out = np.zeros((position_count, position_count))
    #     u_A_in = np.zeros((position_count, position_count))
    #
    #     if len(u_input) == len(set(u_input)):
    #         for i in np.arange(len(u_input) - 1):
    #              u_A_out[i,i:]=1
    #     else:
    #         # print(u_input)
    #         processed = {}
    #         item2idx = {}
    #         for i in range(len(u_input)):
    #             item = u_input[i]
    #             lst = np.where(u_input == item)[0]
    #             item2idx[item] = lst
    #
    #         for i in np.arange(len(u_input) - 1):
    #             # u_lst = np.where(u_input == u_input[i])[0]
    #             u_lst = item2idx[u_input[i]]
    #             u_A_out[i,i:] = 1
    #             for j in np.arange(i,len(u_input),1):
    #                 tuple = (u_input[i],u_input[j])
    #                 if tuple in processed:
    #                     continue
    #                 processed[tuple] = True
    #                 # v_lst = np.where(u_input == u_input[j])[0]
    #                 v_lst = item2idx[u_input[j]]
    #                 for u in u_lst:
    #                     for v in v_lst:
    #                         u_A_out[u][v]  = 1  # 每个结点只计算一次
    #
    #
    #
    #     return u_A_in, u_A_out

    # def construct_graph(self,u_input):
    #     # 自环只加一次
    #     position_count = len(u_input)
    #     u_input = np.array(u_input)
    #     u_A_out = np.zeros((position_count, position_count))
    #     u_A_in = np.zeros((position_count, position_count))
    #
    #
    #     for i in np.arange(len(u_input) - 1):
    #          u_A_out[i,i:]=1
    #     return u_A_in, u_A_out
    # def construct_graph (self,u_input):
    #     position_count = len(u_input)
    #     u_input = np.array(u_input)
    #     u_A_out = np.ones((position_count, position_count))
    #     u_A_in = np.ones((position_count, position_count))
    #
    #     return u_A_in, u_A_out

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

    def load_dataset(self,path, limit):

        dataset = []
        count = 0

        with open(path ,'r') as f:
            for l in f.readlines():
                count+=1
                if count >limit:
                    break

                example = eval(l)
                user_id = example[0]
                item_lst = example[1][:-1]
                target_id = example[7][0]
                length = example[8]-1
                # u_A_in, u_A_out = self.construct_graph(item_lst)
                # u_A_out = example[-1]
                u_A_in = example[-2]
                # u_A_in = np.zeros((len(item_lst),len(item_lst)))
                # u_A_out = np.zeros((len(item_lst),len(item_lst)))
                # u_A_out = np.zeros((1, 1))
                # u_A_in = np.zeros((1, 1))
                # u_A_out = self.construct_graph(item_lst,k=3)
                # u_A_in = self.construct_graph(item_lst)
                # u_A_out = np.zeros_like(u_A_in)


                dataset.append([user_id,item_lst, target_id, length, u_A_in ])

        return dataset

    def load_dataset_normal(self,path, limit):

        dataset = []
        count = 0

        with open(path ,'r') as f:
            for l in f.readlines():
                count+=1
                if count >limit:
                    break

                example = eval(l)
                user_id = example[0]
                item_lst = example[1][:-1]
                target_id = example[7][0]
                length = example[8]-1
                u_A_out = np.zeros((1, 1))
                u_A_in = np.zeros((1, 1))


                dataset.append([user_id,item_lst, target_id, length, u_A_in, u_A_out])


        return dataset

    def read_train_dev_test(self):

        train_limit = 10000000
        test_limit = 20000
        dev_limit = 20000
        # train_limit = 2000
        # test_limit = 2000
        # dev_limit = 2000

        self.train_set  = self.load_dataset(self.train_path,train_limit)
        self.test_set = self.load_dataset(self.test_path, test_limit)
        self.dev_set = self.load_dataset(self.dev_path, dev_limit)

        self.logger.info('train length: %d, dev_length: %d, test_length: %d'%(len(self.train_set),len(self.dev_set), len(self.test_set)))


        return self.train_set, self.test_set, self.dev_set
    def read_train_dev_test_normal(self):

        train_limit = 10000000
        test_limit = 20000
        dev_limit = 20000
        # train_limit = 20000
        # test_limit = 200
        # dev_limit = 200

        self.train_set  = self.load_dataset_normal(self.normal_train_path,train_limit)
        self.test_set = self.load_dataset_normal(self.normal_test_path, test_limit)
        self.dev_set = self.load_dataset_normal(self.normal_dev_path, dev_limit)

        self.logger.info('train length: %d, dev_length: %d, test_length: %d'%(len(self.train_set),len(self.dev_set), len(self.test_set)))


        return self.train_set, self.test_set, self.dev_set
