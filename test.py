# -*- coding: utf-8 -*-
# @Time    : 2020-11-16 09:27
# @Autorchor  : zxl
# @FileName: test.py

import time
import os
import numpy as np
import random
import torch as torch


def seed_torch(seed=2020):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()

if __name__ == "__main__":
	batch_size = 2
	max_len = 3
	num_units = 4
	l = 0
	masks = torch.Tensor(np.array([[[1,0,0],
									[1,1,0],
									[1,1,1]],
								   [[1,1,0],
									[1,1,0],
									[1,1,1]]]))
	x = torch.Tensor(np.array([[[1,2,3,4],
									[1,1,3,3],
									[1,1,1,5]],
								   [[1,1,2,6],
									[1,4,1,4],
									[1,3,1,1]]]))
	cur_masks = masks[:, l,:].view(batch_size, -1, 1)  # batch_size, 50,1
	neighbor_count = torch.sum(cur_masks, axis=1)  # batch_size, 1 
	sum_val = torch.sum((cur_masks * x), axis=1)
	neighbor = sum_val/neighbor_count
	print(cur_masks)
	print(neighbor_count)
	print(sum_val)
	print(neighbor)
