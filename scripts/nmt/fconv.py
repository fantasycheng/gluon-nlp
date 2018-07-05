# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Encoder and decoder used in convolution sequence-to-sequence learning."""
__all__ = ['FConvEncoder', 'FConvDecoder', 'get_fconv_encoder_decoder']

import math
import mxnet as mx
from mxnet.symbol import Dropout, batch_dot, softmax
from mxnet.gluon import nn
from mxnet.gluon.block import Block

try:
    from encoder_decoder import Seq2SeqEncoder, Seq2SeqDecoder
except ImportError:
    from .encoder_decoder import Seq2SeqEncoder, Seq2SeqDecoder

# 用Transformer里的PositionEmbedding替代
def _position_encoding_init(max_length, dim):
    """ Init the sinusoid position encoding table """
    position_enc = np.arange(max_length).reshape((-1, 1)) \
                   / (np.power(10000, (2. / dim) * np.arange(dim).reshape((1, -1))))
    # Apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return position_enc

class FConvEncoder(HybridBlock, Seq2SeqEncoder):
    """Structure of the Convolutional Encoder"""
    def __init__(self, embed_dim=512, convolutions=((512, 3),) * 20, dropout=0.1,
                 max_length=1024, prefix=None, params=None):
        super(FConvEncoder, self).__init__(prefix=prefix, params=params)
        self._dropout = dropout
        self._max_length = max_length
        self._embed_dim = embed_dim
        # self.num_attention_layers = None

        in_channels = convolutions[0][0]
        with self.name_scope():
            self.position_weight = self.params.get_constant('const',
                                                            _position_encoding_init(max_length,
                                                                                    embed_dim))
            self.dropout_layer = nn.Dropout(dropout)
            self.fc1 = nn.Dense(units=in_channels, dropout=dropout,
                                flatten=False, prefix='fc1_')
            self.projections = nn.HybridSequential()
            self.convolutions = nn.HybridSequential()
            for i, (out_channels, kernel_size) in enumerate(convolutions):
                self.projections.add(nn.Dense(out_channels, flatten=False, prex='proj%d_' % i)
                                              if in_channels != out_channels else None)
                if kernel_size % 2 == 1:
                    padding_l = padding_r = kernel_size // 2
                else:
                    padding_l = kernel_size // 2 - 1
                    padding_r = kernel_size // 2
                #这里的Conv1D现只支持‘NCW’layout，且不支持dropout
                self.convolutions.add(nn.Conv1D(out_channels * 2, kernel_size,
                                                padding=(padding_l, padding_r),
                                                prefix='conv%d_' % i))
                in_channels = out_channels
            self.fc2 = nn.Dense(units=embed_dim, flatten=False, prefix='fc2_')

    def __call__(self, inputs, states=None):
        return super(FConvEncoder, self).__call__(inputs, states)

    def forward(self, inputs, states=None, steps=None): 
        length = inputs.shape[1]
        steps = mx.nd.arange(length, ctx=inputs.context)
        if states is None:
            states = [steps]
        else:
            states.append(steps)
        step_output, additional_outputs =\
            super(FConvEncoder, self).forward(inputs, states)
        return step_output, additional_outputs
    
    def hybrid_forward(self, F, inputs, states=None, position_weight=None):
        # add sinusoidal postion embedding temporarily
        if states is not None:
            steps = states[-1]
            # Positional Encoding
            inputs = F.broadcast_add(inputs, F.expand_dims(F.Embedding(steps, position_weight,
                                                                       self._max_length,
                                                                       self._embed_dim), axis=0))

        #Need implement PositionEmbedding in future
        x = self.dropout_layer(inputs)
        input_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> B x C x T
        x = F.swapaxes(x, 1, 2)

        # temporal convolutions
        for proj, conv in zip(self.projections, self.convolutions):
            residual = x if proj is None else proj(x)
            x = self.dropout_layer(x)
            x = conv(x)
            x = self.dropout_layer(x)
            x = glu(x, axis=2)
            x = (x + residual) * math.sqrt(0.5)
        
        # B x C x T -> B x T x C
        x = x.swapaxes(1, 2)

        # project back to size of embedding
        x = self.fc2(x)

        # scale gradients (this only affects backward, not forward)
        # x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(0.5)
        return x, y

class FConvDecoder(Seq2SeqDecoder):
    """Convolutional decoder"""
    def __init__(self, num_embeddings, embed_dict=512, out_embed_dim=256, dropout=0.1, 
                 convolutions=((512, 3),) * 20, attention=True, share_embed=False,
                 prefix=None, params=None):
        super(FConvDecoder, self).__init__(prefix=None, params=None)
        # self.register_buffer('version', torch.Tensor([2]))
        self.dropout = dropout

        in_channels = convolutions[0][0]
        if isinstance(attention, bool):
            attention = [attention] * len(convolutions)
        if not isinstance(attention, list) or len(attention) != len(convolutions):
            raise ValueError("Attention is expected to be a list of booleans of '
                             'length equal to the number of layers.")
        
        with self.name_scope():
            self.fc1 = nn.Dense(units=in_channels, dropout=dropout,
                                flatten=False, prefix='fc1_')
            self.projections = nn.HybridSequential()
            self.convolutions = nn.HybridSequential()
            self.attentions = nn.HybridSequential()
            for i, (out_channels, kernel_size) in enumerate(convolutions):
                self.projections.add(nn.Dense(out_channels, flatten=False, prefix='proj%d_' % i)
                                              if in_channels != out_channels else None)
                self.convolutions.add(nn.Conv1D(out_channels * 2, kernel_size,
                                                padding=kernel_size - 1, prefix='conv%d_' % i))
                self.attentions.add(AttentionLayer(out_channels, embed_dim)
                                                   if attention[i] else None)
                in_channels = out_channels
            self.fc2 = nn.Dense(units=out_embed_dim, flatten=False, prefix='fc2_')
            if share_embed:
                pass
            else:
                self.fc3 = nn.Dense(units=num_embeddings, dropout=dropout)

    def __call__(self, prev_output_tokens, encoder_out, incremental_state=None):
        return super(FConvDecoder, self).__call__(prev_output_tokens, encoder_out)
    
    def forward(self, inputs, encoder_out, incremental_state=None):
        encoder_a, encoder_b = _split_encoder_out(encoder_out, incremental_state)

        x = Dropout(inputs, p=self.dropout, training=self.training)
        target_embedding = x

        x = self.fc1(x)


    
    def _split_encoder_out(self, encoder_out, incremental_state):
        encoder_a, encoder_b = encoder_out
        return encoder_a, encoder_b

def glu(inputs, axis=-1):
    if axis >= len(inputs.shape) or axis < - len(inputs.shape):
        raise RuntimeError("%d index out of range" % (axis))
    d = inputs.shape[axis]
    if d % 2 == 1:
        raise RuntimeError("Inputs size in axis %d must be even" % axis)
    A = inputs.slice_axis(axis=axis, begin=0, end=d//2)
    B = inputs.slice_axis(axis=axis, begin=d//2, end=None)

    return A * B.sigmoid()

def Embedding(num_embeddings, embedding_dim, padding_idx):
    pass

class AttentionLayer(Block):
    def __init__(self, conv_channels, embed_dim):
        super().__init__()
        # projects from output of convolution to embedding dimension
        self.in_projection = nn.Dense(embed_dim, flatten=False, in_units=conv_channels,
                                      prefix='attention_in_proj_')
        # projects from embedding dimension to convolution size
        self.out_projection = nn.Dense(conv_channels, flatten=False, in_unist=embed_dim,
                                       prefix='attention_out_proj_')

        # self.bmm = bmm if bmm is not None else torch.bmm

    def forward(self, x, target_embedding, encoder_out):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
        # BTC x BCT = BTT
        x = batch_dot(x, encoder_out[0])

        # softmax over last dim
        sz = x.shape()
        x = softmax(x.reshape(sz[0] * sz[1], sz[2]), axis=1)
        x = x.reshape(*sz)
        # attn_scores BTT
        attn_scores = x

        # x BTC
        x = batch_dot(x, encoder_out[1])

        # scale attention output
        s = encoder_out[1].shape[1]
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores