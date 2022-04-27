# -*- coding: utf-8 -*-
# @Time    : 2021-05-17 10:39
# @Author  : zxl
# @FileName: stargnn.py


"""
参考 2020 CIKM 文章 Star Graph Neural Networks for Session-based Recommendation 内容实现
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from model.modules.abstract_recommender import SequentialRecommender


class GNN(nn.Module):
    r"""Graph neural networks are well-suited for session-based recommendation,
    because it can automatically extract features of session graphs with considerations of rich node connections.
    """

    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.embedding_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = nn.Parameter(torch.Tensor(self.embedding_size))
        self.b_ioh = nn.Parameter(torch.Tensor(self.embedding_size))

        self.linear_edge_in = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_edge_out = nn.Linear(self.embedding_size, self.embedding_size, bias=True)

    def GNNCell(self, A, hidden):
        r"""Obtain latent vectors of nodes via graph neural networks.

        Args:
            A(torch.FloatTensor):The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]

            hidden(torch.FloatTensor):The item node embedding matrix, shape of
                [batch_size, max_session_len, embedding_size]

        Returns:
            torch.FloatTensor:Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]

        """
        input_in = torch.matmul(A[:, :, :A.size(1)], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.size(1): 2 * A.size(1)], self.linear_edge_out(hidden)) + self.b_ioh
        # [batch_size, max_session_len, embedding_size * 2]
        inputs = torch.cat([input_in, input_out], 2)  # 对应a_s,i

        # gi.size equals to gh.size, shape of [batch_size, max_session_len, embdding_size * 3]
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        # (batch_size, max_session_len, embedding_size)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = (1 - inputgate) * hidden + inputgate * newgate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class StarGNN(SequentialRecommender):
    r"""
    参考2020 CIKM StarGraph 工作实现，在构图基础之上加入star 结点
    Such as: A example of a session sequence(eg:item1, item2, item3, item2, item4) and the connecion matrix A
    """

    def __init__(self, config, item_num ):
        super(StarGNN, self).__init__(config, item_num)

        # load parameters info
        # self.embedding_size = embedding_size
        # self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        # self.n_items = n_items
        # self.n_pos = n_pos
        # self.step = step

        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.step = config['step']
        self.n_pos = config['max_len']

        # define layers and loss
        # item embedding
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.n_pos, self.embedding_size )
        # define layers and loss
        self.gnn = GNN(self.embedding_size, 1)
        self.W_q_one = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.W_k_one = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.W_q_two = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.W_k_two = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.W_g = nn.Linear(2*self.embedding_size, self.embedding_size, bias=True)
        self.linear_one = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_two = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_three = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_zero = nn.Linear(self.embedding_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.embedding_size * 2, self.embedding_size, bias=True)
        # parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            # weight.data.normal_(-0.5, 0.5)


    def get_pos_emb(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        position_embedding = self.position_embedding(position_ids)
        # reverse_position_embedding = self.reverse_position_embedding(reverse_pos_id)
        return position_embedding

    def forward(self, items, alias_inputs, A, item_seq_len ):
        # items = item_seq
        mask = alias_inputs.gt(-1)

        hidden_mask = items.gt(-1)  # 用-1mask的item
        length = torch.sum(hidden_mask, 1).unsqueeze(1)  # B, 1
        items = items.masked_fill(hidden_mask == False,0)

        hidden = self.item_embedding(items)
        # pos_embeddings = self.pos_embedding(poses)

        pos_embeddings = self.get_pos_emb(alias_inputs,item_seq_len)

        h_0 = hidden

        # print(length)
        star_node = torch.sum(hidden, 1)/length # B, d  star node 初始化

        for i in range(self.step):
            hidden, star_node = self.update_item_layer(hidden, star_node.squeeze(), A)

        h_f = self.highway_network(hidden, h_0)

        alias_inputs = alias_inputs.masked_fill(mask == False, 0)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.embedding_size)
        seq_hidden = torch.gather(h_f, dim=1, index=alias_inputs)
        seq_hidden = seq_hidden + pos_embeddings
        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)

        q1 = self.linear_one(seq_hidden)
        q2 = self.linear_two(star_node)
        q3 = self.linear_three(ht).view(ht.size(0), 1, ht.size(1))
        gamma = self.linear_zero(torch.sigmoid(q1 + q2 + q3)) # Batch_size, max_seq_len,1
        a = torch.sum(gamma * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))
        c = seq_output.squeeze()


        # c = ht

        return c



        # l_c = (c/torch.norm(c, dim=-1).unsqueeze(1))
        #
        # l_emb = self.item_embedding.weight[1:]/torch.norm(self.item_embedding.weight[1:], dim=-1).unsqueeze(1)
        # logits = 12 * torch.matmul(l_c, l_emb.t())
        #
        # # logits = torch.matmul(seq_output, self.item_embedding.weight[1:].transpose(0, 1))
        # return logits

    def calculate_logits(self,  items,alias_inputs,A, item_seq_len):

        seq_output = self.forward(items, alias_inputs, A, item_seq_len )
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))


        # l_c = (seq_output / torch.norm(seq_output, dim=-1).unsqueeze(1))
        #
        # l_emb = self.item_embedding.weight /torch.norm(self.item_embedding.weight , dim=-1).unsqueeze(1)
        # logits = 12 * torch.matmul(l_c, l_emb.t())




        return logits


    def update_item_layer(self, h_last, star_node_last, A):
        hidden = self.gnn(A, h_last)
        q_one = self.W_q_one(hidden)  # B, L, d
        k_one = self.W_k_one(star_node_last)  # B, d
        alpha_i = torch.bmm(q_one, k_one.unsqueeze(2))/math.sqrt(self.embedding_size)  # B, L, 1
        new_h = (1 - alpha_i) * hidden + alpha_i * star_node_last.unsqueeze(1)  # B, L, d

        q_two = self.W_q_two(star_node_last)
        k_two = self.W_k_two(new_h)
        beta = torch.softmax(torch.bmm(k_two, q_two.unsqueeze(2))/math.sqrt(self.embedding_size), 1)  # B, L, 1
        new_star_node = torch.bmm(beta.transpose(1, 2), new_h)  # B, 1, d

        return new_h, new_star_node

    def highway_network(self, hidden, h_0):
        g = torch.sigmoid(self.W_g(torch.cat((h_0, hidden), 2)))  # B,L,d
        h_f = g*h_0 + (1-g) * hidden
        return h_f

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the spexific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
