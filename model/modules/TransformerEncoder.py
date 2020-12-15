import torch
import math
import copy


class TransformerEncoder(torch.nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(self,
                 n_layers,
                 n_heads,
                 hidden_size,
                 inner_size,
                 hidden_dropout_prob,
                 attn_dropout_prob,
                 sa_activation='selu',
                 ffn_activation='relu'):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob,
                                 sa_activation=sa_activation, ffn_activation=ffn_activation)
        self.layer = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, queries,keys, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TrandformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer layers' output,
            otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        all_attention_weights = []
        # hidden_states = keys
        for layer_module in self.layer:
            queries, attention_weights = layer_module(queries,keys, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(queries)
                all_attention_weights.append(attention_weights)
        if not output_all_encoded_layers:
            all_encoder_layers.append(queries)
            all_attention_weights.append(all_attention_weights)
        return all_encoder_layers


class TransformerLayer(torch.nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        input (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        output (torch.Tensor): the output of the transformer layer

    """

    def __init__(self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob,
                 sa_activation="selu", ffn_activation="relu"):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadSelfAttention(n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob,
                                                           activation=sa_activation)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, activation=ffn_activation)

    def forward(self, queries,keys, attention_mask):
        attention_output, attention_weights = self.multi_head_attention(queries,keys, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output, attention_weights


class FeedForward(torch.nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        output (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, activation='relu'):
        super(FeedForward, self).__init__()
        self.dense_1 = torch.nn.Linear(hidden_size, inner_size)

        if activation == 'selu':
            self.activation = torch.selu
        elif activation == 'relu':
            self.activation = torch.relu
        else:
            self.activation = torch.relu

        self.dense_2 = torch.nn.Linear(inner_size, hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)

    def forward(self, input):
        output = self.activation(self.dense_1(input))
        output = self.dropout(self.dense_2(output))
        output = self.LayerNorm(output + input)
        return output


class MultiHeadSelfAttention(torch.nn.Module):
    """
    Multi-head self-attention layers

    Args:
        input (torch.Tensor): the input of the layer, (b, seq, dim)
        attention_mask (torch.Tensor): the attention mask for input

    Returns:
        output (torch.Tensor): the output of the layer, (b, seq, dim)
    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, activation="selu"):
        super(MultiHeadSelfAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads))

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_linear = torch.nn.Linear(hidden_size, self.all_head_size)
        self.key_linear = torch.nn.Linear(hidden_size, self.all_head_size)
        # self.value_linear = torch.nn.Linear(hidden_size, self.all_head_size)

        if activation == 'selu':
            self.activation = torch.selu
        elif activation == 'relu':
            self.activation = torch.relu
        else:
            self.activation = torch.relu

        self.attn_dropout = torch.nn.Dropout(attn_dropout_prob)

        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size)
        self.out_dropout = torch.nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (b, seq, head, head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (b, head, seq, head_size)

    def forward(self, queries,keys, attention_mask):
        # input_v = hidden_states
        # q = k = v = hidden_states
        input_v = queries
        q = queries
        k = v = keys

        q = self.activation(self.query_linear(q))  # (b, seq, dim)
        k = self.activation(self.key_linear(k))  # (b, seq, dim)

        q = self.transpose_for_scores(q)  # (b, head, seq, head_size)
        k = self.transpose_for_scores(k)  # (b, head, seq, head_size)
        v = self.transpose_for_scores(v)  # (b, head, seq, head_size)

        attention_scores = q @ k.transpose(-1, -2) / math.sqrt(self.attention_head_size)  # (b, head, seq, seq)
        attention_scores = attention_scores + attention_mask

        attention_weights = self.attn_dropout(torch.softmax(attention_scores, dim=-1))  # (b, head, seq, seq)

        v = attention_weights @ v  # (b, head, seq, head_size)
        v = v.permute(0, 2, 1, 3).contiguous()
        v_shape = v.size()[:-2] + (self.all_head_size,)
        v = v.view(*v_shape)  # (b, seq, dim)
        v = self.dense(v)
        v = self.out_dropout(v)
        v = self.LayerNorm(v + input_v)

        return v, attention_scores
