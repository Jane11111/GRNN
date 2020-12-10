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
from model.modules.layers import TransformerEncoder,StructureAwareTransformerEncoder
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

        self.layer1=nn.Linear(self.embedding_size, self.hidden_size  )
        self.layer2 = nn.Linear(self.embedding_size, self.hidden_size  )

        self.tmp1 = nn.Parameter(torch.Tensor((self.embedding_size)))
        self.tmp2 = nn.Parameter(torch.Tensor((self.embedding_size)))

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
        # semantic_emb = self.layer1(inputs)
        # neighbor_emb = self.layer2(neighbors)
        # new_gate = self.new_gate(torch.cat((semantic_emb,neighbor_emb),1))
        # v8
        semantic_emb = self.relu_activation(self.layer1(inputs))
        neighbor_emb = self.relu_activation(self.layer2(neighbors))
        new_gate = torch.sigmoid(self.new_gate(torch.cat((semantic_emb, neighbor_emb), 1)))

        # v9
        # semantic_emb =  self.relu_activation(self.layer1(torch.cat((inputs,state),1)))
        # neighbor_emb = self.relu_activation(self.layer2(neighbors))
        # new_gate = self.new_gate(torch.cat((semantic_emb, neighbor_emb), 1))

        new_h = u*state  + (1-u) * c*new_gate

        return new_h, new_h
class SelfAtt(nn.Module):
    def __init__(self,embedding_size,n_layers,n_heads,
                 hidden_dropout_prob,att_dropout_prob,
                 hidden_act='gelu', layer_norm_eps = 1e-12):
        super(SelfAtt, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size
        self.inner_size = embedding_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = att_dropout_prob
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps



        self.trm_encoder = TransformerEncoder(n_layers=self.n_layers, n_heads=self.n_heads,
                                              hidden_size=self.hidden_size, inner_size=self.inner_size,
                                              hidden_dropout_prob=self.hidden_dropout_prob,
                                              attn_dropout_prob=self.attn_dropout_prob,
                                              hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)


        self.output_layer = nn.Linear(self.embedding_size*2,self.embedding_size)


    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        # subsequent_mask =  torch.zeros(attn_shape)  # torch.uint8

        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask # 512,1,50,50
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


    def forward(self,item_seq_emb, position_emb ,item_seq):
        """

        :param item_seq_emb: batch_size, max_len, num_units
        :param item_seq: batch_size,
        :return:
        """

        input_emb = item_seq_emb + position_emb
        input_emb = self.dropout(input_emb)


        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb,input_emb,
                                      extended_attention_mask,
                                      output_all_encoded_layers=True)[-1]



        return  trm_output

class ConnectedGNN(nn.Module):
    def __init__(self,embedding_size,n_layers,n_heads,
                 hidden_dropout_prob,att_dropout_prob,
                 hidden_act='gelu', layer_norm_eps = 1e-12):
        super(ConnectedGNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size
        self.inner_size = embedding_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = att_dropout_prob
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps



        self.trm_encoder = TransformerEncoder(n_layers=self.n_layers, n_heads=self.n_heads,
                                              hidden_size=self.hidden_size, inner_size=self.inner_size,
                                              hidden_dropout_prob=self.hidden_dropout_prob,
                                              attn_dropout_prob=self.attn_dropout_prob,
                                              hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)


        self.output_layer = nn.Linear(self.embedding_size*2,self.embedding_size)


    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        # subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask =  torch.zeros(attn_shape)  # torch.uint8

        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask # 512,1,50,50
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_cg_attention_mask(self, adj_mat):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        # attention_mask = (item_seq > 0).long()
        extended_attention_mask = adj_mat.unsqueeze(1) # 512,1,50,50

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask



    def forward(self,item_seq_emb, position_emb,adj_mat):
        """

        :param item_seq_emb: batch_size, max_len, num_units
        :param item_seq: batch_size,
        :return:
        """

        input_emb = item_seq_emb + position_emb
        input_emb = self.dropout(input_emb)

        # in_extended_attention_mask = self.get_cg_attention_mask(adj_in)
        # in_trm_output = self.trm_encoder( input_emb,
        #                               in_extended_attention_mask,
        #                               output_all_encoded_layers=True)[-1]

        out_extended_attention_mask = self.get_cg_attention_mask(adj_mat )
        out_trm_output = self.trm_encoder( input_emb,input_emb,
                                         out_extended_attention_mask,
                                         output_all_encoded_layers=True)[-1]
        # extended_attention_mask = self.get_attention_mask(item_seq)
        # trm_output = self.trm_encoder(input_emb,
        #                               extended_attention_mask,
        #                               output_all_encoded_layers=True)[-1]

        # trm_output = self.output_layer(torch.cat([in_trm_output,out_trm_output],2))

        return out_trm_output
class VallinaAtt(nn.Module):
    def __init__(self,embedding_size,n_layers,n_heads,max_key_length,
                 hidden_dropout_prob,att_dropout_prob,
                 hidden_act='gelu', layer_norm_eps = 1e-12):
        super(VallinaAtt, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size
        self.inner_size = embedding_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_key_length = max_key_length,
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = att_dropout_prob
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps



        self.trm_encoder = StructureAwareTransformerEncoder(
                                              n_layers=self.n_layers, n_heads=self.n_heads,
                                              max_key_length=max_key_length,
                                              hidden_size=self.hidden_size, inner_size=self.inner_size,
                                              hidden_dropout_prob=self.hidden_dropout_prob,
                                              attn_dropout_prob=self.attn_dropout_prob,
                                              hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)


        self.output_layer = nn.Linear(self.embedding_size*2,self.embedding_size)


    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long() # 512,50
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 # 512,1,1,50

        return extended_attention_mask


    def forward(self,keys,queries, keys_graph_emb, queries_graph_emb,position_emb,item_seq ):
        """

        :param item_seq_emb: batch_size, max_len, num_units
        :param item_seq: batch_size,
        :return:
        """
        # keys = keys + position_emb
        # keys = keys+position_emb
        # keys_graph_emb = keys_graph_emb+position_emb

        keys = self.dropout(keys)
        extended_attention_mask = self.get_attention_mask(item_seq )
        trm_output = self.trm_encoder(queries = queries,
                                      keys = keys,
                                      queries_graph_emb=queries_graph_emb,
                                      keys_graph_emb=keys_graph_emb,
                                      attention_mask=extended_attention_mask,
                                      output_all_encoded_layers=True)[-1].squeeze(1)



        return  trm_output




class GNN(nn.Module):

    def __init__(self,embedding_size, graph_emb_size,max_len):

        super(GNN, self).__init__()
        self.embedding_size = embedding_size
        self.max_len = max_len

        self.in_init_layer = nn.Linear(embedding_size,embedding_size,bias=False)
        self.out_init_layer = nn.Linear(embedding_size,embedding_size,bias = False)
        self.self_in_init_layer = nn.Linear(embedding_size, embedding_size, bias=False)
        self.self_out_init_layer = nn.Linear(embedding_size, embedding_size, bias=False)
        self.pos_in_init_layer = nn.Linear(embedding_size, embedding_size, bias=False)
        self.pos_out_init_layer = nn.Linear(embedding_size, embedding_size, bias=False)


        self.in_self_weight = nn.Parameter(torch.Tensor(1, embedding_size))
        self.out_self_weight = nn.Parameter(torch.Tensor(1, embedding_size))

        self.in_adj_weight = nn.Parameter(torch.Tensor(1, embedding_size))
        self.out_adj_weight = nn.Parameter(torch.Tensor(1, embedding_size))

        self.in_pos_weight = nn.Parameter(torch.Tensor(max_len, max_len))
        self.out_pos_weight = nn.Parameter(torch.Tensor(max_len, max_len))

        self.tmp1 = nn.Linear(embedding_size, max_len )
        self.tmp2 = nn.Linear(embedding_size, max_len )

        self.prelu = nn.PReLU(embedding_size)

        self.in_output_layer = nn.Linear(embedding_size*2, embedding_size)
        self.out_output_layer = nn.Linear(embedding_size*2, embedding_size)

        self.gnn_output_layer = nn.Linear(embedding_size*2, embedding_size)

        # self.relu = nn.PReLU()

    def forward(self,init_emb,pos_emb,adj_in, adj_out):
        # adj_in = torch.tensor(adj_in, dtype=torch.float32)
        # adj_out = torch.tensor(adj_out, dtype=torch.float32)
        #
        # self_state_in = self.self_in_init_layer(init_emb)
        # self_state_out = self.self_out_init_layer(init_emb)
        #
        # state_in = self.in_init_layer( init_emb)
        # state_out = self.out_init_layer( init_emb )
        #
        #
        # adj_in_state =  torch.matmul(adj_in, state_in)+self.in_self_weight*self_state_in
        # adj_out_state = torch.matmul(adj_out, state_out)+self.out_self_weight*self_state_out
        #
        # pos_in_state = torch.matmul(self.in_pos_weight,   state_in)
        # pos_out_state = torch.matmul(self.out_pos_weight, state_out)
        #
        #
        # adj_vec_in = self.in_output_layer(
        #     torch.cat([state_in,pos_in_state  ], axis=2))
        # # adj_vec_in = self.prelu(adj_vec_in.view(-1,self.embedding_size)).view(-1,self.max_len,self.embedding_size)
        #
        # adj_vec_out = self.out_output_layer(
        #     torch.cat([state_out,pos_out_state  ], axis=2))
        # # adj_vec_out = self.prelu(adj_vec_out.view(-1,self.embedding_size)).view(-1,self.max_len,self.embedding_size)
        #
        # adj_vec = torch.cat([adj_vec_in, adj_vec_out], axis=-1)
        #
        # adj_vec = self.gnn_output_layer(adj_vec)

        adj_vec = torch.matmul(self.in_pos_weight, init_emb)


        return adj_vec





class GRNN(SequentialRecommender):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, config, item_num):
        super(GRNN, self).__init__(config, item_num)

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

    def forward(self, item_seq, adj_in, item_seq_len):


        position_embedding,reverse_position_embedding = self.get_pos_emb(item_seq,item_seq_len)

        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)

        """
        GraphEmb
        """
        gnn_inputs = item_seq_emb_dropout
        graph_outputs = self.gnn_layers(item_seq_emb=gnn_inputs,
                                 position_emb=position_embedding,
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

        hybrid_preference =   gru_intent + graph_intent


        return hybrid_preference

    def calculate_logits(self, item_seq, item_seq_len, adj_in ):


        seq_output = self.forward(item_seq, adj_in,   item_seq_len)
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return logits


class GRNN_only_graph(SequentialRecommender):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, config, item_num):
        super(GRNN_only_graph, self).__init__(config, item_num)

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
        self.gnn_hidden_dropout_prob = config['gnn_hidden_dropout_prob']
        self.gnn_att_dropout_prob = config['gnn_att_dropout_prob']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.agg_layer = config['agg_layer']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        # position embedding
        self.position_embedding = nn.Embedding(self.max_seq_length+1, self.hidden_size)
        self.reverse_position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)

        self.cg_gnn = ConnectedGNN(embedding_size=self.embedding_size ,
                               n_layers=self.agg_layer,
                               n_heads=1,
                               hidden_dropout_prob=self.gnn_hidden_dropout_prob,
                               att_dropout_prob=self.gnn_att_dropout_prob )

        # GRU
        self.gru_layers_cg = ModifiedGRU(
            input_size=self.embedding_size + self.embedding_size,
            hidden_size=self.hidden_size,
        ).to(config['device'])

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

    def forward(self, item_seq, adj_in, item_seq_len):


        position_embedding,reverse_position_embedding = self.get_pos_emb(item_seq,item_seq_len)

        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)

        """
        GraphEmb
        """

        gnn_inputs = item_seq_emb_dropout
        # adj_graph_seq_emb = self.adj_gnn(item_seq_emb=gnn_inputs,
        #                          position_emb=position_embedding,
        #                          adj_mat = adj_in)

        cg_graph_seq_emb = self.cg_gnn(item_seq_emb=gnn_inputs,
                                 position_emb=position_embedding,
                                 adj_mat = adj_in)

        gru_inputs_cg = torch.cat((item_seq_emb_dropout,cg_graph_seq_emb ),2)
        gru_output_cg, _ = self.gru_layers_cg(gru_inputs_cg)

        graph_intent = self.gather_indexes(cg_graph_seq_emb, item_seq_len-1)

        hybrid_preference = graph_intent


        return hybrid_preference

    def calculate_logits(self, item_seq, item_seq_len, adj_in ):



        seq_output = self.forward(item_seq, adj_in,   item_seq_len)
        # pos_items = interaction[self.POS_ITEM_ID]
        # # self.loss_type = 'CE'
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        # loss = self.loss_fct(logits, pos_items)
        return logits




class GRNN_weak_order(SequentialRecommender):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, config, item_num):
        super(GRNN_weak_order, self).__init__(config, item_num)

        # load parameters info
        self.embedding_size = config['embedding_size']
        # self.graph_emb_size = config['graph_emb_size']

        self.hidden_size = config['hidden_size']
        self.step = config['step']
        self.loss_type = config['loss_type']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']
        self.device = config['device']
        self.gnn_hidden_dropout_prob = config['gnn_hidden_dropout_prob']
        self.gnn_att_dropout_prob = config['gnn_att_dropout_prob']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.agg_layer = config['agg_layer']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        # position embedding
        self.position_embedding = nn.Embedding(self.max_seq_length+1, self.hidden_size)
        self.reverse_position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)

        # GNN
        # self.adj_gnn = GNN(self.embedding_size, self.embedding_size,self.max_seq_length)
        # self.adj_gnn = ConnectedGNN(embedding_size=self.embedding_size,
        #                       n_layers=1,
        #                       n_heads=1,
        #                       hidden_dropout_prob=self.gnn_hidden_dropout_prob,
        #                       att_dropout_prob=self.gnn_att_dropout_prob)
        self.cg_gnn = ConnectedGNN(embedding_size=self.embedding_size ,
                               n_layers=self.agg_layer,
                               n_heads=1,
                               hidden_dropout_prob=self.gnn_hidden_dropout_prob,
                               att_dropout_prob=self.gnn_att_dropout_prob )

        # GRU
        self.gru_layers_cg = GRU_weak_order(
            input_size=self.embedding_size + self.embedding_size,
            hidden_size=self.hidden_size,
        ).to(config['device'])

        self.simple_gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )


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

    def forward(self, item_seq, adj_in, item_seq_len):


        position_embedding,reverse_position_embedding = self.get_pos_emb(item_seq,item_seq_len)

        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)

        """
        GraphEmb
        """

        gnn_inputs = item_seq_emb_dropout

        graph_outputs = self.cg_gnn(item_seq_emb=gnn_inputs,
                                 position_emb=reverse_position_embedding,
                                 adj_mat = adj_in)
        """
        GRUEmb
        """
        gru_inputs  = torch.cat((item_seq_emb_dropout,graph_outputs ),2)
        gru_outputs , _ = self.gru_layers_cg(gru_inputs )
        gru_outputs  = self.dense(gru_outputs)



        """
        READOUT FUNCTION
        """
        gru_intent = self.gather_indexes(gru_outputs, item_seq_len - 1)
        graph_intent = self.gather_indexes(graph_outputs, item_seq_len - 1)

        hybrid_preference =  gru_intent + graph_intent


        return hybrid_preference

    def calculate_logits(self, item_seq, item_seq_len, adj_in ):



        seq_output = self.forward(item_seq, adj_in,   item_seq_len)
        # pos_items = interaction[self.POS_ITEM_ID]
        # # self.loss_type = 'CE'
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        # loss = self.loss_fct(logits, pos_items)
        return logits



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


class GRU_weak_order(nn.Module):

    def __init__(self,input_size, hidden_size):

        super(GRU_weak_order,self).__init__()

        self.cell = ModifiedGRUCell(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.single_size = int(0.5*input_size)

    def forward(self,x):

        batch_size, length = x.size(0), x.size(1)

        outputs = []


        for l in range(length):
            if l == 0:
                state = torch.zeros(batch_size, self.hidden_size,device = x.device )
            else:
                #把前l个emb的avg作为state
                state = torch.mean(x[:,:l,:self.single_size],axis=1)
            output, state = self.cell(x[:,l,:],state)

            outputs.append(output)
        return torch.stack(outputs,1), state


