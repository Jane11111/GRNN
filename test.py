# -*- coding: utf-8 -*-
# @Time    : 2020-11-16 09:27
# @Author  : zxl
# @FileName: test.py

import time
import os
import numpy as np
import random
import torch as th


def seed_torch(seed=2020):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	th.manual_seed(seed)
	th.cuda.manual_seed(seed)
	th.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	th.backends.cudnn.benchmark = False
	th.backends.cudnn.deterministic = True

seed_torch()

if __name__ == "__main__":
    for i in range(10):
        a = random.random()
        print(a)
