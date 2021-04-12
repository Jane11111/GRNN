# -*- coding: utf-8 -*-
# @Time    : 2020-12-29 10:29
# @Author  : zxl
# @FileName: txt2pt.py


"""
将数据转为二进制文件
"""
import struct
import pickle
import numpy as np

if __name__ == "__main__":

    root = '/data/zxl/MTAM-t2/data/training_testing_data/'

    for dir in ['movie_tv_time_item_based_unidirection']:
        dir_path = root+dir+'/'
        print(dir_path)

        for filename in [ 'train_data_new2.txt', 'test_data_new2.txt', 'dev_data_new2.txt']:
            new_filename = filename[:-4]+''+'.pt'
            i = 0
            w_new = open(dir_path+new_filename,'wb')
            dataset = []

            with open(dir_path+filename,'r') as f:
                for l in f:
                    i+=1
                    print(i)
                    # if i > 5000:
                    #     break
                    example = eval(l)
                    user_id = example[0]
                    item_lst = example[1][:-1]
                    target_id = example[7][0]
                    length = example[8] - 1
                    if len(example) == 9:
                        arr = [user_id, item_lst,target_id, length]
                    else:
                        u_A_in = example[-2]
                        arr = [user_id, item_lst, target_id, length,u_A_in]
                    dataset.append(arr)
                w_new.write(pickle.dumps(dataset))

            w_new.close()

            # file = open(dir_path + new_filename,'rb')
            # dataset = pickle.loads(file.read())
            # print(dataset)
