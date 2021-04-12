# -*- coding: utf-8 -*-
# @Time    : 2020-12-27 18:31
# @Author  : zxl
# @FileName: t_test.py
import scipy.stats as stats
import scipy
import numpy as np


from scipy.stats import f

def f_ref(a,b):
    F = np.var(a) / np.var(b)
    df1 = len(a) - 1
    df2 = len(b) - 1
    p_value = 1 - 2 * abs(0.5 - f.cdf(F, df1, df2))
    return p_value

def read_result(path,limit,k):
    lst = []
    count = 0
    with open(path,'r') as f:
        l = f.readline()
        while l :
            count += 1
            if l[0] == '<' or count > limit:
                break
            lst.append(eval(l)[:k])
            l = f.readline()
    return np.array(lst)
def read_test(path,limit):
    labels = []
    count = 0

    with open(path, 'r') as f:
        for l in f.readlines():
            count += 1
            if count > limit:
                break

            example = eval(l)
            target_id = example[7][0]

            labels.append(target_id)

    return labels

def evaluate_result(pred_lst, target_lst):
    n_samples = len(pred_lst)
    hit = []
    ndcg = []
    mrr = []
    for i in range(n_samples):

        target_id = target_lst[i]
        pred_items = pred_lst[i]
        found = False
        for j in range(len(pred_items)):
            if target_id == pred_items[j]:
                found = True
                hit.append(1)
                mrr.append(1/(j+1))
                ndcg.append(np.log(2)/np.log(j+2))
                break
        if not found:
            hit.append(0)
            mrr.append(0)
            ndcg.append(0)
    return np.array(hit),np.array(ndcg),np.array(mrr)




if __name__ == "__main__":

    root = '/home/zxl/project/GRNN/data/best_result/'
    data_name = 'movielen'

    test_path = '/home/zxl/project/MTAM-t2/data/training_testing_data/' + data_name+ '_time_item_based_unidirection/test_data.txt'

    # grnn_path = root+'GRNN_phone.txt'
    # baseline_path = root+'GRU4Rec_phone.txt'
    #
    # grnn_path = root + 'GRNN_tmall_buy.txt'
    # baseline_path = root + 'GRU4Rec_tmall_buy.txt'

    # grnn_path = root + 'GRNN_tmall_buy.txt'
    # baseline_path = root + 'STAMP_tmall_buy.txt'

    # grnn_path = root + 'GRNN_movie_tv.txt'
    # baseline_path = root + 'NARM_movie_tv.txt'

    # grnn_path = root + 'GRNN_home.txt'
    # baseline_path = root + 'LESSR_fast_home.txt'

    # grnn_path = root + 'GRNN_home.txt'
    # baseline_path = root + 'NARM_home.txt'

    # grnn_path = root + 'GRNN_home.txt'
    # baseline_path = root + 'GRU4Rec_home.txt'

    grnn_path = root + 'GRNN_movielen-2.txt'
    baseline_path = root + 'LESSR_fast_movielen.txt'

    # grnn_path = root + 'GRNN_kindle.txt'
    # baseline_path = root + 'SASRec_kindle.txt'

    # grnn_path = root + 'GRNN_elec.txt'
    # baseline_path = root + 'GRU4Rec_elec.txt'

    limit = 20000

    grnn_lst = read_result(grnn_path,limit,20)
    baseline_lst = read_result(baseline_path,limit,20)
    labels = read_test(test_path,limit)

    for k in [1,5,10,20]:

        cur_grnn_lst = grnn_lst[:,:k]
        cur_baseline_lst = baseline_lst[:,:k]

        grnn_hit, grnn_ndcg, grnn_mrr = evaluate_result(cur_grnn_lst,labels)
        baseline_hit, baseline_ndcg, baseline_mrr = evaluate_result(cur_baseline_lst,labels)

        print('grnn @%d, hit: %.5f, ndcg: %.5f, mrr: %.5f'%(k,np.mean(grnn_hit) ,np.mean(grnn_ndcg) ,np.mean(grnn_mrr) ))
        print('baseline @%d, hit: %.5f, ndcg: %.5f, mrr: %.5f'%(k,np.mean(baseline_hit) ,np.mean(baseline_ndcg) ,np.mean(baseline_mrr) ))

        p_hit = stats.ttest_rel(baseline_hit , grnn_hit)
        p_ndcg = stats.ttest_rel(baseline_ndcg , grnn_ndcg )
        p_mrr = stats.ttest_rel(baseline_mrr , grnn_mrr )


        # p_hit = scipy.stats.kendalltau(baseline_hit, grnn_hit)
        # p_ndcg = scipy.stats.kendalltau(baseline_ndcg, grnn_ndcg)
        # p_mrr = scipy.stats.kendalltau(baseline_mrr, grnn_mrr)

        # p_hit = f_ref(baseline_hit, grnn_hit)
        # p_ndcg = f_ref(baseline_ndcg, grnn_ndcg)
        # p_mrr = f_ref(baseline_mrr, grnn_mrr)

        print(p_hit)
        print(p_ndcg)
        print(p_mrr)
