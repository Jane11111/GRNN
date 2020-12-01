# -*- coding: utf-8 -*-
# @Time    : 2020-11-16 09:27
# @Author  : zxl
# @FileName: test.py

import torch
import numpy as np
import time
from utils.data.preprocess import PrepareData
from logging import getLogger


for dataset in [ 'taobaoapp' ]:

    config= {'dataset':dataset}
    print(dataset)
    logger = getLogger('test')
    prepare_data_model = PrepareData(config,logger )
    prepare_data_model.get_train_test_statisitics()