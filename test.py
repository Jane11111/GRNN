# -*- coding: utf-8 -*-
# @Time    : 2020-11-16 09:27
# @Author  : zxl
# @FileName: test.py

import torch
import numpy as np
import torch.nn.functional as F
import time

a = torch.Tensor([[[1,2,3]],
                  [[4, 5, 6]],])
b = torch.Tensor([[[1],
                   [2],
                   [3]],
                  [[1],
                   [2],
                   [3]],
                  ])
c = a+b
print(a)
print(b)
print(c)