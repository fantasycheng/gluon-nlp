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
__all__ = ['FConvEncoder', 'FConvDecoder']

import math
import numpy as np
import mxnet as mx
from mxnet.symbol import Dropout, batch_dot, softmax
from mxnet.gluon import nn
from mxnet.gluon.block import Block, HybridBlock


try:
    from encoder_decoder import Seq2SeqEncoder, Seq2SeqDecoder
except ImportError:
    from .encoder_decoder import Seq2SeqEncoder, Seq2SeqDecoder

# use sinusoidal position embedding temporarily
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
            self.fc1 = nn.Dense(units=in_channels, flatten=False,
                                in_units=embed_dim, prefix='fc1_')
            self.projections = nn.HybridSequential()
            self.convolutions = nn.HybridSequential()
            for i, (out_channels, kernel_size) in enumerate(convolutions):
                self.projections.add(nn.Dense(out_channels, flatten=False, in_units=in_channels, prefix='proj%d_' % i)
                                              if in_channels != out_channels
                                              else nn.HybridLambda('identity', prefix='identity%d_' % i))

                if kernel_size % 2 == 1:
                    padding = kernel_size // 2
                else:
                    padding = 0
                #这里的Conv1D现只支持‘NCW’layout，且不支持dropout
                self.convolutions.add(nn.Conv1D(out_channels * 2, kernel_size,
                                                padding=padding, in_channels=in_channels,
                                                prefix='conv%d_' % i))
                in_channels = out_channels
            self.fc2 = nn.Dense(units=embed_dim, flatten=False,
                                in_units=in_channels, prefix='fc2_')

    def __call__(self, inputs, states=None):
        return super(FConvEncoder, self).__call__(inputs, states)

    def forward(self, inputs, states=None, steps=None): 
        length = inputs.shape[1]
        steps = mx.nd.arange(length, ctx=inputs.context)
        if states is None:
            states = [steps]
        else:
            states.append(steps)
        (x, y), _ = super(FConvEncoder, self).forward(inputs, states)
        return (x, y), []
    
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
        x = self.dropout_layer(x)

        # # B x T x C -> B x C x T
        # x = F.swapaxes(x, 1, 2)

        # temporal convolutions
        for proj, conv in zip(self.projections, self.convolutions):
            residual = proj(x)
            # B x T x C -> B x C x T
            x = F.swapaxes(x, 1, 2)

            x = self.dropout_layer(x)
            kernel_size = conv._kwargs['kernel'][0]
            if kernel_size % 2 == 1:
                x = conv(x)
            else:
                padding_l = (kernel_size - 1) // 2
                padding_r = kernel_size // 2
                # pad function in mxnet only support 4D or 5D inputs
                # so we have to expand dims here
                x = F.expand_dims(x, axis=0)
                x = F.pad(x, mode='constant', pad_width=(0, 0, 0, 0, 0, 0, padding_l, padding_r))
                x = F.squeeze(x, axis=0)
                x = conv(x)
            x = self.dropout_layer(x)
            #x is a Symbol object, so glu function need to know number of channels
            x = glu(x, conv._channels, axis=1)
            
            # B x C x T -> B x T x C
            x = F.swapaxes(x, 1, 2)
            x = (x + residual) * math.sqrt(0.5)
        
        # # B x C x T -> B x T x C
        # x = x.swapaxes(1, 2)

        # project back to size of embedding
        x = self.fc2(x)

        # scale gradients (this only affects backward, not forward)
        # x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(0.5)
        return (x, y), []

class FConvDecoder(HybridBlock, Seq2SeqDecoder):
    """Convolutional decoder"""
    def __init__(self, num_embeddings, embed_dict=512, out_embed_dim=256, max_length=1024, 
                 convolutions=((512, 3),) * 20, attention=True, output_attention=False,
                 dropout=0.1, share_embed=False, prefix=None, params=None):
        super(FConvDecoder, self).__init__(prefix=None, params=None)
        # self.register_buffer('version', torch.Tensor([2]))
        self._dropout = dropout
        self._convolutions = convolutions
        self._output_attention = output_attention

        in_channels = convolutions[0][0]
        if isinstance(attention, bool):
            attention = [attention] * len(convolutions)
        if not isinstance(attention, list) or len(attention) != len(convolutions):
            raise ValueError('Attention is expected to be a list of booleans of '
                             'length equal to the number of layers.')
        self._is_attentions = attention
        self._num_attn_layers = sum(attention)
        with self.name_scope():
            self.position_weight = self.params.get_constant('const',
                                                            _position_encoding_init(max_length,
                                                                                    units))
            self.fc1 = nn.Dense(units=in_channels, flatten=False, prefix='fc1_')
            self.projections = nn.HybridSequential()
            self.convolutions = nn.HybridSequential()
            self.attentions = nn.HybridSequential()
            for i, (out_channels, kernel_size) in enumerate(convolutions):
                self.projections.add(nn.Dense(out_channels, flatten=False, prefix='proj%d_' % i)
                                              if in_channels != out_channels
                                              else HybridLambda('identity', prefix='proj%d_' % i))
                self.convolutions.add(nn.Conv1D(out_channels * 2, kernel_size,
                                                padding=kernel_size - 1, prefix='conv%d_' % i))
                self.attentions.add(AttentionLayer(out_channels, embed_dim)
                                                   if attention[i]
                                                   else HybridLambda('identity', prefix='attn%d_' % i))
                in_channels = out_channels
            self.fc2 = nn.Dense(units=out_embed_dim, flatten=False, prefix='fc2_')
            if share_embed:
                pass
            else:
                self.fc3 = nn.Dense(units=num_embeddings)
    
    def init_state_from_encoder(self, encoder_outputs):
        mem_keys, mem_values = encoder_outputs
        mem_keys.swapaxes(1, 2)
        return encoder_outputs
    
    def decode_seq(self, inputs, states):
        output, states, additional_outputs = self.forward(inputs, states)
        return output, states, additional_outputs

    def __call__(self, inputs, states):
        return super(FConvDecoder, self).__call__(inputs, states)
    
    def forward(self, step_input, states):  
        if len(states) == 2:
            batch_size, input_dims = step_input.shape[0], step_input.shape[-1]
            incremental_states = []
            in_channels = self._convolutions[0][0]
            for out_channels, kernel_size in self._convolutions:
                incremental_states.append(mx.nd.zeros((batch_size, in_channels, kernel_size-1),
                                                      ctx=step_input.context))
                in_channels = out_channels
            states = [incremental_states] + states

        input_shape = step_input.shape
        if len(input_shape) == 2:
            step_input = mx.nd.expand_dims(step_input, axis=1)
        
        length = step_input.shape[1]
        steps = mx.nd.arange(length, ctx=step_input.context)
        states.append(steps)
        step_output, step_additional_outputs = super(FConvDecoder, self).forward(inputs, states)
        states = states[:-1]
        # If it is in testing, only output the last one
        if len(input_shape) == 2:
            step_output = step_output[:, -1, :]
        return step_output, states, step_additional_outputs

    def hybrid_forward(self, F, inputs, states, position_weight=None):
        incremental_states, mem_keys, mem_values, steps = states
        inputs = F.broadcast_add(inputs, F.expand_dims(F.Embedding(steps, position_weight,
                                                                   self._max_length,
                                                                   self._embed_dim), axis=0))
        x = self.dropout_layer(inputs)
        target_embedding = x
        # project to size of convolution
        x = self.fc1(x)  #x.shape (B, T, convolutions[0][0])
        x = self.dropout_layer(x)

        for i, (proj, conv, attn, is_attn, incr_state) in enumerate(zip(self.projections,
                                                                        self.convolutions,
                                                                        self.attentions,
                                                                        self._is_attentions,
                                                                        incremental_states)):
            residual = proj(x)
            x = self.dropout_layer(x)
            # B x T x C -> B x C x T
            x = F.swapaxes(x, 1, 2)
            x = F.concat(incr_state, x, dims=-1)
            incremental_states[i] = x[:, :, 1:]
            x = conv(x)
            x = self.dropout_layer(x)
            #x is a Symbol object, so glu function need to know number of channels
            x = glu(x, conv._channels, axis=1)
            # B x C x T -> B x T x C
            x = F.swapaxes(x, 1, 2)
            if is_attn:
                x, attn_scores = attn(x, target_embedding, (mem_keys, mem_values))
                if self._output_attention:
                    attn_scores = attn_scores / self._num_attn_layers
                    if avg_attn_score is None:
                        avg_attn_score = attn_scores
                    else:
                        avg_attn_score = avg_attn_score + attn_scores
            
            x = (x + residual) * math.sqrt(0.5)

        # project back to size of embedding
        x = self.fc2(x)
        x = self.dropout_layer(x)
        x = self.fc3(x)

        return x, states, avg_attn_score
        


def glu(inputs, num_channels, axis=-1):
    # if axis >= len(inputs.shape) or axis < - len(inputs.shape):
    #     raise RuntimeError("%d index out of range" % (axis))
    # d = inputs.shape[axis]
    if num_channels % 2 == 1:
        raise RuntimeError("Inputs size in axis %d must be even" % axis)
    A = inputs.slice_axis(axis=axis, begin=0, end=num_channels//2)
    B = inputs.slice_axis(axis=axis, begin=num_channels//2, end=None)

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
