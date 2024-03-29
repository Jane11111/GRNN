# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""
import torch
from torch import nn
from torch.nn.init import xavier_normal_

from model.modules.abstract_recommender import SequentialRecommender
from model.modules.layers import TransformerEncoder


class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implmentation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, item_num):
        super(SASRec, self).__init__(config, item_num)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size'] # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = float(config['layer_norm_eps'])
        self.init_range = float(config['init_range'])

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size , padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(n_layers=self.n_layers, n_heads=self.n_heads,
                                              hidden_size=self.hidden_size, inner_size=self.inner_size,
                                              hidden_dropout_prob=self.hidden_dropout_prob,
                                              attn_dropout_prob=self.attn_dropout_prob,
                                              hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)


        # parameters initialization
        self.apply(self._init_weights)
        print('............different initializing................')


    def _init_weights(self, module):
        """ Initialize the weights """


        for weight in self.parameters():

            range= self.init_range
            weight.data.uniform_(-range, range)
    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64 512，1，1，50
        # mask for the whole sequence
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        # subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        # subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        # subsequent_mask = subsequent_mask.long().to(item_seq.device) # 1，1，50，50
        subsequent_mask = torch.ones(attn_shape).unsqueeze(0).to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        # input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb,input_emb,
                                      extended_attention_mask,
                                      output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output # [B H]

    def calculate_logits(self, item_seq, item_seq_len, adj_in=None, adj_out=None):

        seq_output = self.forward(item_seq, item_seq_len)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))

        return logits
