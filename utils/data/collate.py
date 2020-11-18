# -*- coding: utf-8 -*-
# @Time    : 2020-11-15 16:02
# @Author  : zxl
# @FileName: collate.py

import torch
import numpy as np

def collate_fn(samples):
    # samples = np.array(samples)
    user_id, item_seq, target_id, item_seq_len,  adj_in, adj_out = zip(*samples)

    user_id = torch.tensor(user_id,dtype = torch.long)
    item_seq = torch.tensor(item_seq, dtype=torch.long)
    target_id = torch.tensor(target_id, dtype=torch.long)
    item_seq_len = torch.tensor(item_seq_len, dtype=torch.long)
    adj_in = torch.tensor(adj_in, dtype=torch.long)
    adj_out = torch.tensor(adj_out, dtype=torch.long)

    return user_id, item_seq, target_id, item_seq_len,  adj_in, adj_out




def collate_fn_normal(samples):
    # samples = np.array(samples)
    user_id, item_seq, target_id, item_seq_len  = zip(*samples)

    user_id = torch.tensor(user_id,dtype = torch.long)
    item_seq = torch.tensor(item_seq, dtype=torch.long)
    target_id = torch.tensor(target_id, dtype=torch.long)
    item_seq_len = torch.tensor(item_seq_len, dtype=torch.long)

    return user_id, item_seq, target_id, item_seq_len
