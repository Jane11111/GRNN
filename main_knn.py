import pickle
import numpy as np
from model.baseline.SKNN import SKNN
from model.baseline.STAN import STAN
from model.baseline.VSKNN import VSKNN
import math
import time

# load data [id,x,t,y]

def load_dataset(  path, limit):
    count = 0
    id_lst = []
    session_lst = []
    timestamp_lst = []
    predict_lst = []
    with open(path, 'r') as f:
        for l in f.readlines():
            count += 1
            if count > limit:
                break

            example = eval(l)
            user_id = example[0]
            item_lst = example[1][:-1]
            time_lst = example[3][:-1]
            target_id = example[7][0]

            id_lst.append(user_id)
            session_lst.append(item_lst)
            timestamp_lst.append(time_lst[0])
            predict_lst.append(target_id)


    return id_lst,session_lst,timestamp_lst,predict_lst

def read_train_test_data(data_name):
    root = '/home/zxl/project/MTAM-t2/data/'
    train_path = root + 'training_testing_data/' + data_name + '_time_item_based_unidirection/train_data.txt'
    test_path = root + 'training_testing_data/' + data_name + '_time_item_based_unidirection/test_data.txt'
    #
    train_limit = 10000000
    test_limit = 20000

    # train_limit = 1000
    # test_limit = 200

    train_id,train_session, train_timestamp,train_predict = load_dataset(train_path,train_limit)
    test_id,test_session, test_timestamp, test_predict = load_dataset(test_path,test_limit)

    return train_id,train_session, train_timestamp,train_predict,test_id,test_session, test_timestamp, test_predict

dataname = 'home'

train_id,train_session, train_timestamp,train_predict,\
test_id,test_session, test_timestamp, test_predict = read_train_test_data(dataname)

# train_data = pickle.load(open('datasets/retailrocket/train_session_3.txt', 'rb'))
# # train_data = pickle.load(open('datasets/diginetica/train_session_3.txt', 'rb'))
# # train_data = pickle.load(open('datasets/yoochoose/train_1_64_session_2.txt', 'rb'))
# train_id = train_data[0]
# train_session = train_data[1]
# train_timestamp = train_data[2]
# train_predict = train_data[3]

# test_data = pickle.load(open('datasets/retailrocket/test_session_3.txt', 'rb'))
# # test_data = pickle.load(open('datasets/diginetica/test_session_3.txt', 'rb'))
# # test_data = pickle.load(open('datasets/yoochoose/test_1_64_session_2.txt', 'rb'))
# test_id = test_data[0]
# test_session = test_data[1]
# test_timestamp = test_data[2]
# test_predict = test_data[3]



# 用split后的training data,不会造成影响,最后还是用处理后在cache里的数据训练
# 只需把train_session和train_predict合并
for i, s in enumerate(train_session):
    train_session[i] += [train_predict[i]]

print("training size: %d" % len(train_session))
print("testing size: %d" % len(test_session))
# model = SKNN(session_id=train_id, session=train_session, session_timestamp=train_timestamp, sample_size=0, k=500)
model = VSKNN(session_id=train_id, session=train_session, session_timestamp=train_timestamp, sample_size=0, k=500)

# model = STAN(session_id=train_id, session=train_session, session_timestamp=train_timestamp,
#              sample_size=0, k=500, factor1=True, l1=3.0, factor2=True, l2=20*24*3600, factor3=True, l3=2.54)

testing_size = len(test_session)
# testing_size = 10

R_5 = 0
R_10 = 0
R_20 = 0

MRR_5 = 0
MRR_10 = 0
MRR_20 = 0

NDCG_5 = 0
NDCG_10 = 0
NDCG_20 = 0
for i in range(testing_size):
    if i % 1000 == 0:
        print("%d/%d" % (i, testing_size))
        # print("MRR@20: %f" % (MRR_20 / (i + 1)))

    score = model.predict(session_id=test_id[i], session_items=test_session[i], session_timestamp=test_timestamp[i],
                          k=20)
    # for s in score:
    #     print(s)
    # print(test_predict[i])
    # print("-----------------------------------")
    # print("-----------------------------------")
    items = [x[0] for x in score]
    # if len(items) == 0:
    #     print("!!!")
    if test_predict[i] in items:
        rank = items.index(test_predict[i]) + 1
        # print(rank)
        MRR_20 += 1 / rank
        R_20 += 1
        NDCG_20 += 1 / math.log(rank + 1, 2)

        if rank <= 5:
            MRR_5 += 1 / rank
            R_5 += 1
            NDCG_5 += 1 / math.log(rank + 1, 2)

        if rank <= 10:
            MRR_10 += 1 / rank
            R_10 += 1
            NDCG_10 += 1 / math.log(rank + 1, 2)

MRR_5 = MRR_5 / testing_size
MRR_10 = MRR_10 / testing_size
MRR_20 = MRR_20 / testing_size
R_5 = R_5 / testing_size
R_10 = R_10 / testing_size
R_20 = R_20 / testing_size
NDCG_5 = NDCG_5 / testing_size
NDCG_10 = NDCG_10 / testing_size
NDCG_20 = NDCG_20 / testing_size

print(dataname)

print(' hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f'
                % (R_5, NDCG_5, MRR_5 ,
                   R_10, NDCG_10, MRR_10,
                   R_20, NDCG_20, MRR_20))

