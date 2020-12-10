# -*- coding: utf-8 -*-
# @Time    : 2020-11-19 09:51
# @Author  : zxl
# @FileName: test2.py

import numpy as np
import torch

batch_size = 2
max_len = 3
emb_size = 4



a = torch.tensor(np.ones((batch_size,max_len,emb_size)))
b = torch.mean(a[:,:2,:],axis=1)
print(b)