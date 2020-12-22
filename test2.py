# -*- coding: utf-8 -*-
# @Time    : 2020-11-19 09:51
# @Author  : zxl
# @FileName: test2.py

import numpy as np
import torch

p_lst = [0.9,0.09,0.01]

res = 0

for idx in range(len(p_lst)):
    p = p_lst[idx]
    res -= p*np.log (p )

print(res)