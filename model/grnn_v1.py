# -*- coding: utf-8 -*-
# @Time    : 2020-11-15 11:55
# @Author  : zxl
# @FileName: grnn.py
# -*- coding: utf-8 -*-
# @Time    : 2020-11-12 14:10
# @Author  : zxl
# @FileName: grnn.py

import math

import torch
from torch import nn

from model.modules.abstract_recommender import SequentialRecommender

class ModifiedGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModifiedGRUCell,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = int(input_size/2)

        self.reset_linear = nn.Linear(self.embedding_size  + self.hidden_size,self.hidden_size)
        self.update_linear = nn.Linear(self.embedding_size + self.hidden_size, self.hidden_size)
        self.candidate_linear = nn.Linear(self.embedding_size + self.hidden_size,self.hidden_size)

        self.layer1=nn.Linear(self.embedding_size , self.hidden_size)
        self.layer2 = nn.Linear(self.embedding_size, self.hidden_size)
        self.new_r_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)


        self.new_gate = nn.Linear(self.hidden_size*2, self.hidden_size)

        self.relu_activation = torch.nn.ReLU()

    def forward(self,inputs, state):

        neighbors = inputs[:,-self.embedding_size:]
        inputs = inputs[:,:self.embedding_size]

        gate_inputs = torch.cat((inputs,state),1)

        r = torch.sigmoid(self.reset_linear(gate_inputs))
        u = torch.sigmoid(self.update_linear(gate_inputs))

        # v5
        # new_r = self.new_r_gate(torch.cat((inputs,neighbors),1))
        # new_u = self.new_gate(torch.cat((inputs,neighbors),1))
        # r_state = r*state*new_r

        r_state = r*state

        candidate = self.candidate_linear(torch.cat((inputs,r_state),1))
        c = torch.tanh(candidate)
        # v1
        # new_gate = self.new_gate(torch.cat((inputs,neighbors),1))

        # v2
        # semantic_emb = self.layer1(torch.cat((inputs, state),1))
        # neighbor_emb = self.layer2(neighbors)
        # new_gate = self.new_gate(torch.cat((semantic_emb,neighbor_emb),1))

        # v3
        # semantic_emb = self.relu_activation(self.layer1(torch.cat((inputs, state), 1)))
        # neighbor_emb = self.relu_activation(self.layer2(neighbors))
        # new_gate = torch.sigmoid(self.new_gate(torch.cat((semantic_emb, neighbor_emb), 1)))

        # v4
        # candidate = self.candidate_linear(torch.cat((inputs, neighbors,r_state), 1))
        # c = torch.tanh(candidate)

        # v7
        semantic_emb = self.layer1(inputs)
        neighbor_emb = self.layer2(neighbors)
        new_gate = self.new_gate(torch.cat((semantic_emb,neighbor_emb),1))


        new_h = u*state  + (1-u) * c*new_gate

        return new_h, new_h


class ModifiedGRU(nn.Module):

    def __init__(self,input_size, hidden_size):

        super(ModifiedGRU,self).__init__()

        self.cell = ModifiedGRUCell(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self,x,h=None):

        batch_size, length = x.size(0), x.size(1)

        outputs = []

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size,device = x.device )
        state = h
        for l in range(length):
            output, state = self.cell(x[:,l,:],state)

            outputs.append(output)
        return torch.stack(outputs,1), state

class NaiveGRU(nn.Module):

    def __init__(self,input_size, hidden_size):

        super(NaiveGRU,self).__init__()

        self.cell = ModifiedGRUCell(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self,x):

        batch_size, length = x.size(0), x.size(1)

        outputs = []


        for l in range(length):
            state = torch.zeros(batch_size, self.hidden_size,device = x.device )
            output, state = self.cell(x[:,l,:],state)

            outputs.append(output)
        return torch.stack(outputs,1), state


class GNN(nn.Module):

    def __init__(self,embedding_size, graph_emb_size,max_len):

        super(GNN, self).__init__()

        self.in_init_layer = nn.Linear(embedding_size,graph_emb_size)
        self.out_init_layer = nn.Linear(embedding_size,graph_emb_size)
        self.self_in_init_layer = nn.Linear(embedding_size, embedding_size )
        self.self_out_init_layer = nn.Linear(embedding_size, embedding_size )

        self.in_pos_weight = nn.Parameter(torch.Tensor(max_len, max_len))
        self.out_pos_weight = nn.Parameter(torch.Tensor(max_len, max_len))

        self.in_linear1 = nn.Linear(embedding_size, 1)
        self.out_linear1 = nn.Linear(embedding_size, 1)
        self.in_linear2 = nn.Linear(embedding_size, 1)
        self.out_linear2 = nn.Linear(embedding_size, 1)

        self.in_linear = nn.Linear(embedding_size*2, embedding_size)
        self.out_linear = nn.Linear(embedding_size*2, embedding_size)

        self.in_output_layer = nn.Linear(graph_emb_size*3, graph_emb_size)
        self.out_output_layer = nn.Linear(graph_emb_size*3, graph_emb_size)

        self.in_self_weight = nn.Parameter(torch.Tensor(1, embedding_size))
        self.out_self_weight = nn.Parameter(torch.Tensor(1, embedding_size))

        self.gnn_output_layer = nn.Linear(graph_emb_size*2, graph_emb_size)

        self.relu = nn.ReLU()

    def forward(self,init_emb,adj_in, adj_out):


        fin_state_in = self.in_init_layer(init_emb)
        fin_state_out = self.out_init_layer(init_emb)
        self_state_in = self.self_in_init_layer(init_emb)
        self_state_out = self.self_out_init_layer(init_emb)

        adj_in = torch.tensor(adj_in,dtype=torch.float32)
        adj_out = torch.tensor(adj_out,dtype=torch.float32)

        adj_in_state = torch.matmul(adj_in, fin_state_in)
        adj_out_state = torch.matmul(adj_out, fin_state_out)

        pos_in_state = torch.matmul(self.in_pos_weight,   fin_state_in)
        pos_out_state = torch.matmul(self.out_pos_weight, fin_state_out)

        adj_vec_in = self.in_output_layer(
            torch.cat([fin_state_in, adj_in_state,pos_in_state], axis=2))
        adj_vec_out = self.out_output_layer(
            torch.cat([fin_state_out,adj_out_state,pos_out_state], axis=2))

        adj_vec = torch.cat([adj_vec_in, adj_vec_out], axis=-1)


        adj_vec = self.gnn_output_layer(adj_vec)


        return adj_vec



class GRNN_v1(SequentialRecommender):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, config, item_num):
        super(GRNN_v1, self).__init__(config, item_num)

        # load parameters info
        self.embedding_size = config['embedding_size']
        # self.graph_emb_size = config['graph_emb_size']
        self.layer_norm_eps = float(config['layer_norm_eps'])

        self.hidden_size = config['hidden_size']
        self.step = config['step']
        self.loss_type = config['loss_type']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']
        self.device = config['device']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        # position embedding
        self.position_embedding = nn.Embedding(self.max_seq_length+1, self.hidden_size)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # GNN
        self.gnn = GNN(self.embedding_size, self.embedding_size,self.max_seq_length)

        # GRU
        self.gru_layers = ModifiedGRU(
            input_size=self.embedding_size + self.embedding_size,
            hidden_size=self.hidden_size,
        ).to(config['device'])

        # READOUT
        self.dense = nn.Linear(self.hidden_size , self.embedding_size)

        self.W1 = nn.Linear(self.hidden_size,self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size,self.hidden_size)
        self.r = nn.Parameter(torch.Tensor(self.hidden_size,))
        self.q = nn.Linear(self.hidden_size,1)
        self.output_layer = nn.Linear(self.hidden_size*2, self.hidden_size)

        self.tmp1= nn.Linear(self.hidden_size*2, self.hidden_size)



        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):


        if isinstance(module,nn.GRU) or isinstance(module, ModifiedGRU)  :

            for weight in module.parameters():
                if len(weight.shape) == 2:
                    nn.init.orthogonal_(weight.data)
        else:
            stdv = 1.0 / math.sqrt(self.embedding_size)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)




    def forward(self, item_seq, adj_in,adj_out,item_seq_len):

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)


        new_seq_len = item_seq_len.view(-1, 1)
        reverse_pos_id = new_seq_len - position_ids
        reverse_pos_id = torch.clamp(reverse_pos_id, 0)

        position_embedding = self.position_embedding(reverse_pos_id)


        item_seq_emb = self.item_embedding(item_seq)
        # item_seq_emb = self.LayerNorm(item_seq_emb)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)

        """
        GraphEmb
        """
        #TODO gru添加了embedding
        graph_seq_emb = self.gnn(item_seq_emb_dropout+position_embedding,adj_in,adj_out)
        """
        GRUEmb
        """
        gru_inputs = torch.cat((item_seq_emb_dropout,graph_seq_emb),2)

        gru_output, _ = self.gru_layers(gru_inputs)
        # 最原始版本，是最后一个embedding 过了一层非线性层

        # gru_output = self.dense(  torch.cat((item_seq_emb_dropout,
        #                                      graph_seq_emb ,
        #                                      gru_output  ),2))
        gru_output = self.dense(gru_output)

        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        """
        READOUT
        """
        # READOUT 0

        # x_last = self.gather_indexes(gru_output, item_seq_len - 1)
        #
        # pos_gru_output = self.tmp1(torch.cat((gru_output, position_embedding),2))
        #
        # trans_item_embedding = self.W1(pos_gru_output)
        # trans_last_item_embedding = self.W2(x_last).view(-1,1,self.embedding_size)
        #
        # att_val_inputs = trans_item_embedding+trans_last_item_embedding+self.r
        #
        # att_val = self.q(att_val_inputs).view(-1,self.max_seq_length)
        # att_val = att_val.view(-1,self.max_seq_length,1) # batch_size, max_len,1
        #
        # x_global = torch.mean(att_val* gru_output,1)
        # seq_output = self.output_layer(torch.cat((x_last,x_global),1))

        # READOUT 2

        # gru_output += position_embedding
        #
        # x_last = self.gather_indexes(gru_output, item_seq_len - 1)
        #
        # trans_item_embedding = self.W1(item_seq_emb_dropout+position_embedding)
        # trans_last_item_embedding = self.W2(x_last).view(-1, 1, self.embedding_size)
        #
        # att_val_inputs = trans_item_embedding + trans_last_item_embedding + self.r
        #
        # att_val = self.q(att_val_inputs).view(-1, self.max_seq_length)
        # att_val = F.softmax(att_val, dim=1).view(-1, self.max_seq_length, 1)  # batch_size, max_len,1
        #
        # x_global = torch.mean(att_val * gru_output, 1)
        # seq_output = self.output_layer(torch.cat((x_last, x_global), 1))

        # READOUT 3

        # x_last = self.gather_indexes(gru_output, item_seq_len - 1)
        #
        # x_avg = torch.mean(gru_output,1)
        #
        # seq_output = self.output_layer(torch.cat((x_last, x_avg), 1))

        return seq_output

    def calculate_logits(self, item_seq, item_seq_len, adj_in, adj_out):



        seq_output = self.forward(item_seq, adj_in, adj_out, item_seq_len)
        # pos_items = interaction[self.POS_ITEM_ID]
        # # self.loss_type = 'CE'
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        # loss = self.loss_fct(logits, pos_items)
        return logits


class GRNN_mlp(SequentialRecommender):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, config, item_num):
        super(GRNN_mlp, self).__init__(config, item_num)

        # load parameters info
        self.embedding_size = config['embedding_size']
        # self.graph_emb_size = config['graph_emb_size']
        self.layer_norm_eps = float(config['layer_norm_eps'])

        self.hidden_size = config['hidden_size']
        self.step = config['step']
        self.loss_type = config['loss_type']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']
        self.device = torch.device(config['device'])

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        # position embedding
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # GNN
        self.gnn = GNN(self.embedding_size, self.embedding_size, self.max_seq_length).to(self.device)

        # GRU
        # self.gru_layers = ModifiedGRU(
        #     input_size=self.embedding_size + self.embedding_size,
        #     hidden_size=self.hidden_size,
        # ).to(config['device'])
        # self.gru_layers = nn.GRU(
        #     input_size=self.embedding_size * 2,
        #     hidden_size=self.hidden_size,
        #     num_layers=1,
        #     bias=False,
        #     batch_first=True,
        # )
        self.output_layer = nn.Linear(self.embedding_size*2, self.embedding_size).to(self.device)

        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):

        # if isinstance(module, ModifiedGRU)  :
        #     for weight in module.parameters():
        #         if len(weight.shape) == 2:
        #             nn.init.orthogonal_(weight.data)
        # if isinstance(module, nn.Embedding):
        #     xavier_normal_(module.weight)
        if isinstance(module, nn.GRU) or isinstance(module, ModifiedGRU):
            # xavier_uniform_(self.gru_layers.weight_hh_l0)
            # xavier_uniform_(self.gru_layers.weight_ih_l0)
            for weight in module.parameters():
                if len(weight.shape) == 2:
                    nn.init.orthogonal_(weight.data)
        else:
            stdv = 1.0 / math.sqrt(self.embedding_size)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, adj_in, adj_out, item_seq_len):


        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        new_seq_len = item_seq_len.view(-1, 1)
        reverse_pos_id = new_seq_len - position_ids
        reverse_pos_id = torch.clamp(reverse_pos_id, 0)

        position_embedding = self.position_embedding(reverse_pos_id)

        item_seq_emb = self.item_embedding(item_seq)
        # item_seq_emb = self.LayerNorm(item_seq_emb)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)

        """
        GraphEmb
        """
        # TODO gru添加了embedding
        graph_seq_emb = self.gnn(item_seq_emb_dropout + position_embedding, adj_in, adj_out)
        """
        MLPEmb
        """
        gru_inputs = torch.cat((item_seq_emb_dropout, graph_seq_emb), 2)
        gru_output = self.output_layer(gru_inputs)

        gru_output = self.dense(gru_output)


        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)

        # seq_output = self.LayerNorm(seq_output)

        return seq_output

    def calculate_logits(self, item_seq, item_seq_len, adj_in, adj_out):
        # item_seq = interaction[self.ITEM_SEQ]
        # item_seq_len = interaction[self.ITEM_SEQ_LEN]
        #
        # adj_in = interaction['adj_in']
        # adj_out = interaction['adj_out']

        seq_output = self.forward(item_seq, adj_in, adj_out, item_seq_len)
        # pos_items = interaction[self.POS_ITEM_ID]
        # # self.loss_type = 'CE'
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        # loss = self.loss_fct(logits, pos_items)
        return logits


class GRNN_no_order(SequentialRecommender):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, config, item_num):
        super(GRNN_no_order, self).__init__(config, item_num)


        # load parameters info
        self.embedding_size = config['embedding_size']
        # self.graph_emb_size = config['graph_emb_size']
        self.layer_norm_eps = float(config['layer_norm_eps'])

        self.hidden_size = config['hidden_size']
        self.step = config['step']
        self.loss_type = config['loss_type']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']
        self.device = config['device']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        # position embedding
        self.position_embedding = nn.Embedding(self.max_seq_length+1, self.hidden_size)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # GNN
        self.gnn = GNN(self.embedding_size, self.embedding_size,self.max_seq_length)

        # GRU
        self.gru_layers = NaiveGRU(
            input_size=self.embedding_size + self.embedding_size,
            hidden_size=self.hidden_size,
        ).to(config['device'])
        # self.gru_layers = nn.GRU(
        #     input_size=self.embedding_size * 2,
        #     hidden_size=self.hidden_size,
        #     num_layers=1,
        #     bias=False,
        #     batch_first=True,
        # )

        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):

        # if isinstance(module, ModifiedGRU)  :
        #     for weight in module.parameters():
        #         if len(weight.shape) == 2:
        #             nn.init.orthogonal_(weight.data)
        # if isinstance(module, nn.Embedding):
        #     xavier_normal_(module.weight)
        if isinstance(module,nn.GRU) or isinstance(module, ModifiedGRU)  :
            # xavier_uniform_(self.gru_layers.weight_hh_l0)
            # xavier_uniform_(self.gru_layers.weight_ih_l0)
            for weight in module.parameters():
                if len(weight.shape) == 2:
                    nn.init.orthogonal_(weight.data)
        else:
            stdv = 1.0 / math.sqrt(self.embedding_size)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)



    def forward(self, item_seq, adj_in,adj_out,item_seq_len):

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)


        new_seq_len = item_seq_len.view(-1, 1)
        reverse_pos_id = new_seq_len - position_ids
        reverse_pos_id = torch.clamp(reverse_pos_id, 0)

        position_embedding = self.position_embedding(reverse_pos_id)


        item_seq_emb = self.item_embedding(item_seq)
        # item_seq_emb = self.LayerNorm(item_seq_emb)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)

        """
        GraphEmb
        """
        #TODO gru添加了embedding
        graph_seq_emb = self.gnn(item_seq_emb_dropout+position_embedding,adj_in,adj_out)
        """
        GRUEmb
        """
        gru_inputs = torch.cat((item_seq_emb_dropout,graph_seq_emb),2)

        gru_output, _ = self.gru_layers(gru_inputs)

        gru_output = self.dense(gru_output)



        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)

        # seq_output = self.LayerNorm(seq_output)

        return seq_output

    def calculate_logits(self, item_seq, item_seq_len, adj_in, adj_out):


        seq_output = self.forward(item_seq, adj_in, adj_out, item_seq_len)
        # pos_items = interaction[self.POS_ITEM_ID]
        # # self.loss_type = 'CE'
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        # loss = self.loss_fct(logits, pos_items)
        return logits
