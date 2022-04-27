# -*- coding: utf-8 -*-
# @Time    : 2021-04-30 10:43
# @Author  : zxl
# @FileName: grnn_ablation.py

import math

import torch
from torch import nn
from model.modules.layers import TransformerEncoder,StructureAwareTransformerEncoder
from model.modules.abstract_recommender import SequentialRecommender
from model.grnn import ConnectedGNN,ModifiedGRU


class GatedAgg(nn.Module):

    def __init__(self,embedding_size):
        super(GatedAgg, self).__init__()
        self.embedding_size = embedding_size

        self.in_linear = nn.Linear(embedding_size,embedding_size)
        self.out_linear = nn.Linear(embedding_size,embedding_size)

        self.final_linear = nn.Linear(2*embedding_size,embedding_size)


    def forward(self,item_seq_emb,adj_in,adj_out):



        in_emb = torch.matmul(adj_in,item_seq_emb)
        in_emb = self.in_linear(in_emb)
        out_emb = torch.matmul(adj_out,item_seq_emb)
        out_emb = self.out_linear(out_emb)

        emb = self.final_linear(torch.cat((in_emb,out_emb),2))
        return emb

class WGATdAgg(nn.Module):

    def __init__(self,embedding_size,n_heads ):
        super(WGATdAgg, self).__init__()
        self.embedding_size = embedding_size

        self.n_heads = n_heads

        self.shared_linear = nn.Linear(embedding_size,embedding_size*self.n_heads)
        self.att1_linear = nn.Parameter(torch.Tensor(1,embedding_size*self.n_heads))
        self.att2_linear = nn.Parameter(torch.Tensor(1,embedding_size*self.n_heads))
        self.att3_weight = nn.Parameter(torch.Tensor(1,self.n_heads))

        self.leaklyrelu_fun = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.softmax_fun = nn.Softmax(2)
        self.relu_fun = nn.ReLU()



    def forward(self,item_seq_emb,adj_in,adj_out):

        batch_size,L,d = item_seq_emb.shape

        item_seq_emb = self.shared_linear(item_seq_emb) # B, L, d*head_num

        part1 = (item_seq_emb*self.att1_linear).reshape(batch_size,L, self.n_heads,d) # B,L, head_num,d
        part1 = torch.sum(part1,dim=3).permute(0,2,1).unsqueeze(3) # B,head_num,L,1
        part2 = (item_seq_emb*self.att2_linear).reshape(batch_size,L,self.n_heads,d) # B,L, head_num,d
        part2 = torch.sum(part2,dim =3).permute(0,2,1).unsqueeze(3).permute(0,1,3,2) # B,head_num,1,L

        part3 = (self.att3_weight*adj_out.unsqueeze(3)).permute(0,3,1,2) # B,head_num,L,L

        alpha = part1+part2+part3 # B,head_num,L,L

        alpha = self.leaklyrelu_fun(alpha)

        masks = torch.zeros_like(adj_out) # B,L,L
        masks[adj_out==0] = (-2**31+1)
        masks = masks.reshape(batch_size,1,L,L) # B,1,L,L

        alpha=alpha+masks

        alpha = self.softmax_fun(alpha) # B,head_num,L,L

        item_seq_emb = item_seq_emb.reshape(batch_size,self.n_heads,L,d) # B,head_num,L,d

        weighted_emb = torch.matmul(alpha,item_seq_emb) # B,head_num,L,d
        weighted_emb = torch.mean(weighted_emb,dim=1,keepdim = False) # B,L,d

        output = self.relu_fun(weighted_emb)

        return output

class gated_my_update(SequentialRecommender):


    def __init__(self, config, item_num):
        super(gated_my_update, self).__init__(config, item_num)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.device = config['device']
        # self.gnn_hidden_dropout_prob = config['gnn_hidden_dropout_prob']
        # self.gnn_att_dropout_prob = config['gnn_att_dropout_prob']
        # self.agg_layer = config['agg_layer']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        # position embedding
        self.position_embedding = nn.Embedding(self.max_seq_length+1, self.hidden_size)
        self.reverse_position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)

        self.gnn_layers = GatedAgg(embedding_size=self.embedding_size)

        # GRU
        self.gru_layers= ModifiedGRU(
            input_size=self.embedding_size + self.embedding_size,
            hidden_size=self.hidden_size,
        ).to(config['device'])

        self.dense = nn.Linear(self.hidden_size  , self.embedding_size )
        # parameters initialization
        self.apply(self._init_weights)
        print('............initializing................')



    def _init_weights(self, module):


        if isinstance(module,nn.GRU) or isinstance(module, ModifiedGRU)  :

            for weight in module.parameters():
                if len(weight.shape) == 2:
                    nn.init.orthogonal_(weight.data)
        else:
            stdv = 1.0 / math.sqrt(self.embedding_size)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)


    def get_pos_emb(self,item_seq,item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        new_seq_len = item_seq_len.view(-1, 1)
        reverse_pos_id = new_seq_len - position_ids
        reverse_pos_id = torch.clamp(reverse_pos_id, 0)

        position_embedding = self.position_embedding(position_ids)
        reverse_position_embedding = self.reverse_position_embedding(reverse_pos_id)
        return position_embedding, reverse_position_embedding

    def forward(self, item_seq,item_seq_len, adj_in, adj_out):


        # position_embedding,reverse_position_embedding = self.get_pos_emb(item_seq,item_seq_len)

        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)

        """
        **********************FINAL MODEL***********************
        """
        """
        GraphEmb
        """
        gnn_inputs = item_seq_emb_dropout
        graph_outputs = self.gnn_layers(item_seq_emb=gnn_inputs,
                                        adj_in = adj_in,
                                        adj_out = adj_out)
        """
        GRUEmb
        """
        gru_inputs = torch.cat((item_seq_emb_dropout,graph_outputs ),2)
        gru_outputs, _ = self.gru_layers(gru_inputs)

        gru_outputs = self.dense(gru_outputs)


        """
        READOUT FUNCTION
        """
        gru_intent = self.gather_indexes(gru_outputs, item_seq_len - 1)
        graph_intent = self.gather_indexes(graph_outputs, item_seq_len - 1)

        hybrid_preference = gru_intent + graph_intent


        return hybrid_preference

    def calculate_logits(self, item_seq, item_seq_len, adj_in ,adj_out):


        seq_output = self.forward(item_seq, item_seq_len, adj_in,adj_out)
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return logits



class wgat_my_update(SequentialRecommender):


    def __init__(self, config, item_num):
        super(wgat_my_update, self).__init__(config, item_num)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.n_heads = config['n_heads']
        # self.step = config['step']
        # self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']
        self.device = config['device']
        # self.gnn_hidden_dropout_prob = config['gnn_hidden_dropout_prob']
        # self.gnn_att_dropout_prob = config['gnn_att_dropout_prob']
        # self.agg_layer = config['agg_layer']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        # position embedding
        self.position_embedding = nn.Embedding(self.max_seq_length+1, self.hidden_size)
        self.reverse_position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)

        self.gnn_layers = WGATdAgg(embedding_size=self.embedding_size,n_heads=self.n_heads)

        # GRU
        self.gru_layers= ModifiedGRU(
            input_size=self.embedding_size + self.embedding_size,
            hidden_size=self.hidden_size,
        ).to(config['device'])

        self.dense = nn.Linear(self.hidden_size  , self.embedding_size )
        # parameters initialization
        self.apply(self._init_weights)
        print('............initializing................')



    def _init_weights(self, module):


        if isinstance(module,nn.GRU) or isinstance(module, ModifiedGRU)  :

            for weight in module.parameters():
                if len(weight.shape) == 2:
                    nn.init.orthogonal_(weight.data)
        else:
            stdv = 1.0 / math.sqrt(self.embedding_size)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)


    def get_pos_emb(self,item_seq,item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        new_seq_len = item_seq_len.view(-1, 1)
        reverse_pos_id = new_seq_len - position_ids
        reverse_pos_id = torch.clamp(reverse_pos_id, 0)

        position_embedding = self.position_embedding(position_ids)
        reverse_position_embedding = self.reverse_position_embedding(reverse_pos_id)
        return position_embedding, reverse_position_embedding

    def forward(self, item_seq,item_seq_len, adj_in, adj_out):


        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)

        """
        **********************FINAL MODEL***********************
        """
        """
        GraphEmb
        """
        gnn_inputs = item_seq_emb_dropout
        graph_outputs = self.gnn_layers(item_seq_emb=gnn_inputs,
                                        adj_in = adj_in,
                                        adj_out = adj_out)
        """
        GRUEmb
        """
        gru_inputs = torch.cat((item_seq_emb_dropout,graph_outputs ),2)
        gru_outputs, _ = self.gru_layers(gru_inputs)

        gru_outputs = self.dense(gru_outputs)


        """
        READOUT FUNCTION
        """
        gru_intent = self.gather_indexes(gru_outputs, item_seq_len - 1)
        graph_intent = self.gather_indexes(graph_outputs, item_seq_len - 1)

        hybrid_preference = gru_intent + graph_intent


        return hybrid_preference

    def calculate_logits(self, item_seq, item_seq_len, adj_in ,adj_out):


        seq_output = self.forward(item_seq, item_seq_len, adj_in,adj_out)
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return logits


class grnn_no_duplication(SequentialRecommender):


    def __init__(self, config, item_num):
        super(grnn_no_duplication, self).__init__(config, item_num)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.step = config['step']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']
        self.device = config['device']
        self.gnn_hidden_dropout_prob = config['gnn_hidden_dropout_prob']
        self.gnn_att_dropout_prob = config['gnn_att_dropout_prob']
        self.agg_layer = config['agg_layer']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        # position embedding
        self.position_embedding = nn.Embedding(self.max_seq_length+1, self.hidden_size)
        self.reverse_position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)

        self.gnn_layers = ConnectedGNN(embedding_size=self.embedding_size ,
                               n_layers=self.agg_layer,
                               n_heads=1,
                               hidden_dropout_prob=self.gnn_hidden_dropout_prob,
                               att_dropout_prob=self.gnn_att_dropout_prob )

        # GRU
        self.gru_layers= ModifiedGRU(
            input_size=self.embedding_size + self.embedding_size,
            hidden_size=self.hidden_size,
        ).to(config['device'])

        self.dense = nn.Linear(self.hidden_size  , self.embedding_size )
        # parameters initialization
        self.apply(self._init_weights)
        print('............initializing................')



    def _init_weights(self, module):


        if isinstance(module,nn.GRU) or isinstance(module, ModifiedGRU)  :

            for weight in module.parameters():
                if len(weight.shape) == 2:
                    nn.init.orthogonal_(weight.data)
        else:
            stdv = 1.0 / math.sqrt(self.embedding_size)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)


    def get_pos_emb(self,item_seq,item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        new_seq_len = item_seq_len.view(-1, 1)
        reverse_pos_id = new_seq_len - position_ids
        reverse_pos_id = torch.clamp(reverse_pos_id, 0)

        position_embedding = self.position_embedding(position_ids)
        reverse_position_embedding = self.reverse_position_embedding(reverse_pos_id)
        return position_embedding, reverse_position_embedding

    def forward(self, item_seq, item_seq_len, adj_in,adj_out):


        # position_embedding,reverse_position_embedding = self.get_pos_emb(item_seq,item_seq_len)

        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)

        position_embedding = torch.zeros_like(item_seq_emb).to(item_seq.device)


        """
        **********************FINAL MODEL***********************
        """
        """
        GraphEmb
        """
        gnn_inputs = item_seq_emb_dropout
        graph_outputs = self.gnn_layers(item_seq_emb=gnn_inputs,
                                 position_emb= position_embedding,
                                 adj_mat = adj_in)
        """
        GRUEmb
        """
        gru_inputs = torch.cat((item_seq_emb_dropout,graph_outputs ),2)
        gru_outputs, _ = self.gru_layers(gru_inputs)

        gru_outputs = self.dense(gru_outputs)


        """
        READOUT FUNCTION
        """
        gru_intent = self.gather_indexes(gru_outputs, item_seq_len - 1)
        graph_intent = self.gather_indexes(graph_outputs, item_seq_len - 1)

        hybrid_preference = gru_intent + graph_intent


        return hybrid_preference

    def calculate_logits(self, item_seq, item_seq_len, adj_in,adj_out ):


        seq_output = self.forward(item_seq, item_seq_len, adj_in,adj_out)
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return logits
