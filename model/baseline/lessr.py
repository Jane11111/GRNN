# -*- coding: utf-8 -*-
# @Time    : 2020-11-22 12:02
# @Author  : zxl
# @FileName: lessr.py

import numpy as np
import torch as th
from torch import nn
import dgl
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from collections import Counter
from model.modules.abstract_recommender import SequentialRecommender

print(dgl.__path__)

class DGLGRAPH():

    def __init__(self,device):
        self.device = device

    def construct_graph(self,item_seq, item_seq_len):
        seqs = []
        for i in range(len(item_seq_len )):
            l = item_seq_len[i]
            seqs.append( item_seq[i,:l] )


        res = []
        for seq_to_graph in [ self.seq_to_eop_multigraph,  self.seq_to_shortcut_graph]:
            graphs = list(map(seq_to_graph, seqs))
            bg = dgl.batch(graphs)
            res.append(bg)
        return res

    def label_last(self, g, last_nid):
        is_last = np.zeros(g.number_of_nodes(), dtype=np.int32)
        is_last[last_nid] = 1
        g.ndata['last'] = th.tensor(is_last).to(self.device)
        return g

    def seq_to_eop_multigraph(self, seq):
        items =  np.unique(seq  )
        iid2nid = {iid: i for i, iid in enumerate(items)}
        num_nodes = len(items)

        g = dgl.DGLGraph().to(self.device)
        g.add_nodes(num_nodes)
        g.ndata['iid'] = th.tensor(items).to(self.device)

        if len(seq) > 1:
            seq_nid = [iid2nid[iid] for iid in seq]
            src = seq_nid[:-1]
            dst = seq_nid[1:]
            # edges are added in the order of their occurrences.
            g.add_edges(src, dst)

        self.label_last(g, iid2nid[seq[-1]])
        return g

    def seq_to_shortcut_graph(self,seq):
        items = np.unique(seq)
        iid2nid = {iid: i for i, iid in enumerate(items)}
        num_nodes = len(items)

        g = dgl.DGLGraph().to(self.device)
        g.add_nodes(num_nodes)
        g.ndata['iid'] = th.tensor(items).to(self.device)

        seq_nid = [iid2nid[iid] for iid in seq]
        counter = Counter(
            [(seq_nid[i], seq_nid[j]) for i in range(len(seq)) for j in range(i, len(seq))]
        )
        edges = counter.keys()
        src, dst = zip(*edges)
        # edges are added in the order of their first occurrences.
        g.add_edges(src, dst)

        return g



class EOPA(nn.Module):
    def __init__(self, input_dim, output_dim, activation=None, batch_norm=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.gru = nn.GRU(input_dim, input_dim, batch_first=True)
        self.fc_self = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neigh = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation

    def reducer(self, nodes):
        m = nodes.mailbox['m']
        _, hn = self.gru(m)  # hn: (1, batch_size, d)
        return {'neigh': hn.squeeze(0)}

    def forward(self, mg, feat):
        with mg.local_scope():
            if self.batch_norm is not None:
                feat = self.batch_norm(feat)
            mg.ndata['ft'] = feat
            if mg.number_of_edges() > 0:
                mg.update_all(fn.copy_u('ft', 'm'), self.reducer)
                neigh = mg.ndata['neigh']
                rst = self.fc_self(feat) + self.fc_neigh(neigh)
            else:
                rst = self.fc_self(feat)
            if self.activation is not None:
                rst = self.activation(rst)
            return rst


class SGAT(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, activation=None, batch_norm=True
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.fc_q = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, output_dim, bias=False)
        self.attn_e = nn.Linear(hidden_dim, 1, bias=False)
        self.activation = activation

    def forward(self, sg, feat):
        with sg.local_scope():
            if self.batch_norm is not None:
                feat = self.batch_norm(feat)
            q = self.fc_q(feat)
            k = self.fc_k(feat)
            v = self.fc_v(feat)
            sg.ndata.update({'q': q, 'k': k, 'v': v})
            sg.apply_edges(fn.u_add_v('q', 'k', 'e'))
            e = self.attn_e(th.sigmoid(sg.edata['e']))
            sg.edata['a'] = edge_softmax(sg, e)
            sg.update_all(fn.u_mul_e('v', 'a', 'm'), fn.sum('m', 'ft'))
            rst = sg.ndata['ft']
            if self.activation is not None:
                rst = self.activation(rst)
            return rst


class AttnReadout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_norm=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        self.attn_e = nn.Linear(hidden_dim, 1, bias=False)
        if output_dim != input_dim:
            self.fc_out = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.fc_out = nn.Identity()

    def forward(self, g, feat, last_nodes):
        with g.local_scope():
            if self.batch_norm is not None:
                feat = self.batch_norm(feat)
            feat_u = self.fc_u(feat)
            feat_v = self.fc_v(feat[last_nodes])
            feat_v = dgl.broadcast_nodes(g, feat_v)
            g.ndata['e'] = self.attn_e(th.sigmoid(feat_u + feat_v))
            alpha = dgl.softmax_nodes(g, 'e')
            g.ndata['w'] = feat * alpha
            rst = dgl.sum_nodes(g, 'w')
            rst = self.fc_out(rst)
            return rst


class LESSR(SequentialRecommender):
    def __init__(self, config, item_num):
        super(LESSR, self).__init__(config, item_num)
        num_items = item_num
        embedding_dim = config['embedding_size']
        num_layers = config['step']
        self.device = config['device']

        self.item_embedding = nn.Embedding(num_items, embedding_dim, max_norm=1)
        self.indices = nn.Parameter(
            th.arange(num_items, dtype=th.long), requires_grad=False
        )
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        input_dim = embedding_dim
        for i in range(num_layers):
            if i % 2 == 0:
                layer = EOPA(
                    input_dim,
                    embedding_dim,
                    activation=nn.PReLU(embedding_dim),
                    batch_norm=True,
                )
            else:
                layer = SGAT(
                    input_dim,
                    embedding_dim,
                    embedding_dim,
                    activation=nn.PReLU(embedding_dim),
                    batch_norm=True,
                )
            input_dim += embedding_dim
            self.layers.append(layer)
        self.readout = AttnReadout(
            input_dim, embedding_dim, embedding_dim, batch_norm=True
        )
        input_dim += embedding_dim
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.fc_sr = nn.Linear(input_dim, embedding_dim, bias=False)

    def forward(self, mg, sg):
        iid = mg.ndata['iid']
        feat = self.item_embedding(iid)# 无序
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                out = layer(mg, feat)
            else:
                out = layer(sg, feat)
            feat = th.cat([out, feat], dim=1)
        last_nodes = mg.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        # seq_idx = list([lambda nodes: nodes.data['iid']])
        sr_g = self.readout(mg, feat, last_nodes)
        sr_l = feat[last_nodes]
        sr = th.cat([sr_l, sr_g], dim=1)
        sr = self.fc_sr(self.batch_norm(sr))
        # logits = sr @ self.item_embedding(self.indices).t()
        # return logits
        return  sr


    def calculate_logits(self, item_seq, item_seq_len):

        item_seq = item_seq.to('cpu').numpy()
        item_seq_len = item_seq_len.to('cpu').numpy()

        graph_model = DGLGRAPH(self.device)
        mg,sg = graph_model.construct_graph(item_seq,item_seq_len)
        # mg = mg.to(self.device)
        # sg = sg.to(self.device)


        seq_output = self.forward(mg, sg)

        test_item_emb = self.item_embedding.weight
        logits = th.matmul(seq_output, test_item_emb.transpose(0, 1))

        return logits


