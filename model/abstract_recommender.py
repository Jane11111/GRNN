# -*- coding: utf-8 -*-
# @Time    : 2020-11-15 11:57
# @Author  : zxl
# @FileName: abstract_recommender.py

from torch import nn
import numpy as np


class AbstractRecommender(nn.Module):
    r"""Base class for all models
    """

    # def calculate_loss(self, interaction):
    #
    #     raise NotImplementedError

    # def predict(self, interaction):
    #
    #     raise NotImplementedError

    # def calculate_logits(self, interaction):
    #
    #     raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class SequentialRecommender(AbstractRecommender):
    """
    This is a abstract sequential recommender. All the sequential model should implement This class.
    """

    def __init__(self, config, item_num):
        super(SequentialRecommender, self).__init__()

        self.n_items = item_num
        self.max_seq_length = config['max_len']
        self.config = config

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the spexific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
