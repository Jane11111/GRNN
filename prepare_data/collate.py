# -*- coding: utf-8 -*-
# @Time    : 2020-11-15 16:02
# @Author  : zxl
# @FileName: collate.py

import torch
import numpy as np
import dgl
from collections import Counter

def label_last(g, last_nid):
    is_last = np.zeros(g.number_of_nodes(), dtype=np.int32)
    is_last[last_nid] = 1
    g.ndata['last'] = torch.tensor(is_last)
    return g


def seq_to_eop_multigraph(seq):
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.ndata['iid'] = torch.tensor(items)

    if len(seq) > 1:
        seq_nid = [iid2nid[iid] for iid in seq]
        src = seq_nid[:-1]
        dst = seq_nid[1:]
        # edges are added in the order of their occurrences.
        g.add_edges(src, dst)

    label_last(g, iid2nid[seq[-1]])
    return g


def seq_to_shortcut_graph(seq):
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.ndata['iid'] = torch.tensor(items)

    seq_nid = [iid2nid[iid] for iid in seq]
    counter = Counter(
        [(seq_nid[i], seq_nid[j]) for i in range(len(seq)) for j in range(i, len(seq))]
    )
    edges = counter.keys()
    src, dst = zip(*edges)
    # edges are added in the order of their first occurrences.
    g.add_edges(src, dst)

    return g



def collate_fn(samples):
    # samples = np.array(samples)
    user_id, item_seq, target_id, item_seq_len ,adj_in  = zip(*samples)

    user_id = torch.tensor(user_id,dtype = torch.long)
    item_seq = torch.tensor(item_seq, dtype=torch.long)
    target_id = torch.tensor(target_id, dtype=torch.long)
    item_seq_len = torch.tensor(item_seq_len, dtype=torch.long)
    adj_in = torch.tensor(adj_in, dtype=torch.long)
    # adj_out = torch.tensor(adj_out, dtype=torch.long)

    return user_id, item_seq, target_id, item_seq_len,  adj_in
def collate_gnn_fn(samples):
    # samples = np.array(samples)
    user_id, item_seq, target_id, item_seq_len ,adj_in,adj_out  = zip(*samples)

    user_id = torch.tensor(user_id,dtype = torch.long)
    item_seq = torch.tensor(item_seq, dtype=torch.long)
    target_id = torch.tensor(target_id, dtype=torch.long)
    item_seq_len = torch.tensor(item_seq_len, dtype=torch.long)
    adj_in = torch.tensor(adj_in, dtype=torch.float)
    adj_out = torch.tensor(adj_out, dtype=torch.float)

    return user_id, item_seq, target_id, item_seq_len,  adj_in,adj_out

def collate_stargnn_fn(samples):
    # samples = np.array(samples)
    user_id, item_seq, target_id, item_seq_len ,items,alias_inputs,A  = zip(*samples)

    user_id = torch.tensor(user_id,dtype = torch.long)
    item_seq = torch.tensor(item_seq, dtype=torch.long)
    target_id = torch.tensor(target_id, dtype=torch.long)
    item_seq_len = torch.tensor(item_seq_len, dtype=torch.long)
    items = torch.tensor(items,dtype = torch.long)
    alias_inputs = torch.tensor(alias_inputs,dtype=torch.long)
    A = torch.tensor(A, dtype=torch.float)

    return user_id, item_seq, target_id, item_seq_len, items,alias_inputs,A



def collate_fn_normal(samples):
    # samples = np.array(samples)
    user_id, item_seq, target_id, item_seq_len  = zip(*samples)

    user_id = torch.tensor(user_id,dtype = torch.long)
    item_seq = torch.tensor(item_seq, dtype=torch.long)
    target_id = torch.tensor(target_id, dtype=torch.long)
    item_seq_len = torch.tensor(item_seq_len, dtype=torch.long)

    return user_id, item_seq, target_id, item_seq_len

def collate_fn_lessr(samples):
    # samples = np.array(samples)
    user_id, item_seq, target_id, item_seq_len  = zip(*samples)

    res_graphs = []
    for seq_to_graph in [seq_to_eop_multigraph, seq_to_shortcut_graph]:
        graphs = list(map(seq_to_graph, item_seq))
        bg = dgl.batch(graphs)
        res_graphs.append(bg)
    target_id = torch.tensor(target_id,dtype=torch.long)




    return res_graphs[0],res_graphs[1] ,target_id
