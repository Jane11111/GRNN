# -*- coding: utf-8 -*-
# @Time    : 2020-11-15 11:57
# @Author  : zxl
# @FileName: dataset.py

import numpy as np
from torch.utils.data import Dataset

class PaddedDataset(Dataset):
    def __init__(self, sessions, max_len):
        self.sessions = sessions
        self.max_len = max_len


    def __getitem__(self, idx):
        seq = self.sessions[idx]


        user_id = seq[0]
        item_lst = seq[1]
        target_id = seq[2]
        length = seq[3]
        u_A_in = seq[4]
        # u_A_out = seq[5]

        seq_padding_size = [0, self.max_len-length]
        matrix_padding_size = [(0,self.max_len -length), (0, self.max_len - length)]

        padded_item_lst = np.pad(item_lst, seq_padding_size,'constant')
        padded_u_A_in = np.pad(u_A_in, matrix_padding_size,'constant',constant_values=(0,0))
        # padded_u_A_out = np.pad(u_A_out, matrix_padding_size, 'constant', constant_values=(0, 0))
        # padded_u_A_in = np.array(u_A_in)
        # padded_u_A_out = np.array(u_A_out)

        return  [user_id, padded_item_lst, target_id, length,
                               padded_u_A_in ]
        # return seq

    def __len__(self):
        return len(self.sessions)

class PaddedDatasetNormal(Dataset):
    def __init__(self, sessions, max_len):
        self.sessions = sessions
        self.max_len = max_len


    def __getitem__(self, idx):
        seq = self.sessions[idx]


        # for seq in seqs:
        #     [user_id,item_lst, target_id, length, u_A_in, u_A_out]
        user_id = seq[0]
        item_lst = seq[1]
        target_id = seq[2]
        length = seq[3]


        seq_padding_size = [0, self.max_len-length]

        padded_item_lst = np.pad(item_lst, seq_padding_size,'constant')


        return  [user_id, padded_item_lst, target_id, length ]
        # return seq

    def __len__(self):
        return len(self.sessions)

class  DatasetLessr(Dataset):
    def __init__(self, sessions, max_len):
        self.sessions = sessions
        self.max_len = max_len


    def __getitem__(self, idx):
        seq = self.sessions[idx]


        # for seq in seqs:
        #     [user_id,item_lst, target_id, length, u_A_in, u_A_out]
        user_id = seq[0]
        item_lst = seq[1]
        target_id = seq[2]
        length = seq[3]


        # seq_padding_size = [0, self.max_len-length]

        # padded_item_lst = np.pad(item_lst, seq_padding_size,'constant')


        return  [user_id, item_lst, target_id, length ]
        # return seq

    def __len__(self):
        return len(self.sessions)
