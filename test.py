# -*- coding: utf-8 -*-
# @Time    : 2020-11-16 09:27
# @Autorchor  : zxl
# @FileName: test.py

import time
import os
import numpy as np
import random
import torch as torch


def construct_graph_no_duplication(self, u_input):
	# 自环只加一次

	position_count = len(u_input)
	u_input = np.array(u_input)
	# u_A_out = np.zeros(shape=(position_count, position_count), dtype=np.int)
	u_A_in = np.zeros(shape=(position_count, position_count), dtype=np.int)

	item2singleidx = {}
	item2idx = {}

	for i in range(len(u_input)):
		item2singleidx[u_input[i]] = i
		if u_input[i] not in item2idx:
			item2idx[u_input[i]] = []
		item2idx[u_input[i]].append(i)
	processed_items = {}
	for i in range(len(u_input) - 1, -1, -1):
		if u_input[i] in processed_items:
			continue
		processed_items[u_input[i]] = True
		for u in item2idx[u_input[i]]:
			for j in range(i):
				v_idx = item2singleidx[u_input[j]]
				u_A_in[u][v_idx] = 1

	u_A_in = u_A_in.tolist()
	return u_A_in, u_A_in

if __name__ == "__main__":

	a = [1,2,3]
	seq_padding_size = [0,7]
	padded_alias_inputs = np.pad(a, seq_padding_size, 'constant', constant_values=(-1, -1))
	print(padded_alias_inputs)
