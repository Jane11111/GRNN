# -*- coding: utf-8 -*-
# @Time    : 2020-12-04 15:25
# @Author  : zxl
# @FileName: main_pop.py

import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch as th
np.random.seed(1234)

def map_process(origin_data):
    """
    Map origin_data to one-hot-coding except time.

    """
    item_le = preprocessing.LabelEncoder()
    user_le = preprocessing.LabelEncoder()
    cat_le = preprocessing.LabelEncoder()

    # get item id list
    item_id = item_le.fit_transform( origin_data["item_id"].tolist())

    # get user id list
    user_id = user_le.fit_transform( origin_data["user_id"].tolist())

    # get category id list
    cat_id = cat_le.fit_transform (origin_data["cat_id"].tolist())


    # _key:keyµƒ¡–±Ì£¨_map:keyµƒ¡–±Ìº”±‡∫≈
    origin_data['item_id'] = item_id
    origin_data['user_id'] = user_id
    origin_data['cat_id'] = cat_id

    origin_data = origin_data.sort_values(['user_id','time_stamp'])
    origin_data = origin_data.reset_index(drop=True)

    return origin_data


def evaluate(labels, topk):
    l = len(labels)
    log2 = th.log(th.tensor(2.))
    # TODO
    labels = th.tensor(np.array(labels).reshape((-1,1)))
    topk = th.tensor(np.array(topk))

    hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
    r_ranks = 1 / hit_ranks.to(th.float32)
    hit = hit_ranks.numel()
    mrr = r_ranks.sum()
    ndcg = (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

    hit/=l
    mrr/=l
    ndcg/=l
    return hit, ndcg, mrr


def get_topk(user_lst,most_pop_lst, user_pop_dic,k):

    t_pop = []
    p_pop = []
    for user_id in user_lst:
        t_pop.append(most_pop_lst[:k])

        cur_user_pop = user_pop_dic[user_id]
        cur_user_pop = cur_user_pop[:min(k,len(cur_user_pop))]
        while len(cur_user_pop)< k:
            cur_user_pop.append(-1)
        p_pop.append(cur_user_pop)

    return t_pop, p_pop


if __name__ == "__main__":

    dataset = 'phone'

    root = '/home/zxl/project/MTAM-t2/data/'
    origin_path = root+'orgin_data/'+dataset+'.csv'

    test_path = root + 'training_testing_data/' + dataset \
                + '_time_item_based_unidirection/test_data.txt'


    """
    find t_pop, p_pop
    """

    df = pd.read_csv(origin_path)
    df = map_process(df)
    print(df.shape)

    # 删除最后两个item：dev&test
    lst = []
    user_df = df.groupby('user_id')
    for user_id, group_df in user_df:
        group_df = group_df.sort_values('time_stamp',ascending=True)
        new_df = group_df
        lst.append(new_df)

    print('behavior count: '+str(df.shape))
    df = pd.concat(lst,axis=0)
    user_count = len(lst)
    print('user count: %d'%user_count)
    print('train count: %d'%(df.shape[0]-user_count*3))



    # item_df = df.groupby('item_id').count().sort_values('user_id',ascending=False)
    # top_pop = list(item_df.index.values)
    top_pop = df.groupby(by=['item_id'],as_index=False)['user_id'].count()
    top_pop = top_pop.sort_values(['user_id'],ascending=False)['item_id'].tolist()

    user_pop = {}
    # for user_id, group_df in df.groupby('user_id'):
    #     item_df = group_df.groupby('item_id').count().sort_values('user_id',ascending=False)
    #     user_pop[user_id] = list(item_df.index.values)
    """
    test
    """
    user_lst = []
    target_lst = []
    count = 0
    with open(test_path,'r') as f:
        l = f.readline()
        while l:
            example = eval(l)
            user_id = example[0]
            target_id = example[7][0]
            item_lst = example[1][:-1]

            # item_df = pd.DataFrame({'item_id':item_lst,'user_id':user_id})
            # item_df = item_df.groupby('item_id').count().sort_values('user_id', ascending=False)
            # user_pop[user_id] = list(item_df.index.values)

            item_freq_dic = {}
            for iid in item_lst:
                if iid not in item_freq_dic:
                    item_freq_dic[iid] = 0
                item_freq_dic[iid]+=1
            item_freq_dic = sorted(item_freq_dic.items(),key=lambda item: item[1],reverse=True)
            user_pop[user_id] = [x[0] for x in item_freq_dic]



            user_lst.append(user_id)
            target_lst.append(target_id)
            l = f.readline()
            count+=1
            if count >=20000 :
                break

    t_pop5, p_pop5 = get_topk(user_lst,top_pop,user_pop,5)
    t_pop10, p_pop10 = get_topk(user_lst, top_pop, user_pop, 10)
    t_pop20, p_pop20 = get_topk(user_lst, top_pop, user_pop, 20)

    """
    evaluate
    """



    print('-----------t_top---------------')
    hit_5, ndcg_5, mrr_5 = evaluate(target_lst, t_pop5)
    hit_10, ndcg_10, mrr_10 = evaluate(target_lst, t_pop10)
    hit_20, ndcg_20, mrr_20 = evaluate(target_lst, t_pop20)
    print(' hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                % (hit_5, ndcg_5, mrr_5 ,
                   hit_10, ndcg_10, mrr_10,
                   hit_20, ndcg_20, mrr_20))

    print('-----------p_pop--------------')

    hit_5, ndcg_5, mrr_5 = evaluate(target_lst, p_pop5)
    hit_10, ndcg_10, mrr_10 = evaluate(target_lst, p_pop10)
    hit_20, ndcg_20, mrr_20 = evaluate(target_lst, p_pop20)
    print(' hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                % (hit_5, ndcg_5, mrr_5 ,
                   hit_10, ndcg_10, mrr_10,
                   hit_20, ndcg_20, mrr_20))

