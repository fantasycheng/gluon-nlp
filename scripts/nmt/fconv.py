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
    from encoder_decoder import Seq2SeqEncoder, Seq2SeqDecoder, _get_attention_cell
except ImportError:
    from .encoder_decoder import Seq2SeqEncoder, Seq2SeqDecoder, _get_attention_cell

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
                 max_length=1024, normalization_constant=0.5, prefix=None, params=None):
        super(FConvEncoder, self).__init__(prefix=prefix, params=params)
        self._dropout = dropout
        self._max_length = max_length
        self._embed_dim = embed_dim
        self._normalization_constant = normalization_constant
        # self.num_attention_layers = None

        convolutions = extend_conv_spec(convolutions)
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
            self.residuals = []

            layers_in_channels = [in_channels]
            for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
                if residual == 0:
                    residual_dim = out_channels
                else:
                    residual_dim = layers_in_channels[-residual]
                self.projections.add(nn.Dense(out_channels, flatten=False, in_units=residual_dim, prefix='proj%d_' % i)
                                              if residual_dim != out_channels
                                              else nn.HybridLambda('identity', prefix='identity%d_' % i))

                if kernel_size % 2 == 1:
                    padding = kernel_size // 2
                else:
                    padding = 0
                #Conv1D only supports ‘BCT’ layout for now, and don't support dropout
                self.convolutions.add(nn.Conv1D(out_channels * 2, kernel_size,
                                                padding=padding, in_channels=in_channels,
                                                prefix='conv%d_' % i))
                self.residuals.append(residual)
                in_channels = out_channels
                layers_in_channels.append(out_channels)
            self.fc2 = nn.Dense(units=embed_dim, flatten=False,
                                in_units=in_channels, prefix='fc2_')

    def __call__(self, inputs, states=None, valid_length=None):
        return super(FConvEncoder, self).__call__(inputs, states, valid_length)

    def forward(self, inputs, states=None, valid_length=None, steps=None): 
        length = inputs.shape[1]
        # if valid_length is not None:
        #     mask = mx.nd.broadcast_lesser(
        #         mx.nd.arange(length, ctx=inputs.context).reshape((1, -1)),
        #         valid_length.reshape((-1, 1)))
        #     mask = mx.nd.broadcast_axes(mx.nd.expand_dims(mask, axis=1), axis=1, size=length)
        #     if states is None:
        #         states.append(mask)
        #     else:
        #         states.append(mask)
        steps = mx.nd.arange(length, ctx=inputs.context)
        if states is None:
            states = [steps]
        else:
            states.append(steps)
        
        (x, y), _ = super(FConvEncoder, self).forward(inputs, states, valid_length)
        return (x, y), []
    
    def hybrid_forward(self, F, inputs, states=None, valid_length=None, position_weight=None):
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
        residuals = [x]
        for proj, conv, res_layer in zip(self.projections, self.convolutions, self.residuals):
            if res_layer > 0:
                residual = proj(residuals[-res_layer])
            else:
                residual = None 
            # B x T x C -> B x C x T
            if valid_length is not None:
                x = F.SequenceMask(x, sequence_length=valid_length,
                                   use_sequence_length=True, axis=1)
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
            if residual is not None:
                x = (x + residual) * math.sqrt(self._normalization_constant)
            residuals.append(x)
        
        # # B x C x T -> B x T x C
        # x = x.swapaxes(1, 2)

        # project back to size of embedding
        x = self.fc2(x)
        if valid_length is not None:
            x = F.SequenceMask(x, sequence_length=valid_length,
                               use_sequence_length=True, axis=1)

        # scale gradients (this only affects backward, not forward)
        # x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(self._normalization_constant)
        return (x, y), []

class FConvDecoder(HybridBlock, Seq2SeqDecoder):
    """Convolutional decoder"""
    def __init__(self, embed_dim=512, out_embed_dim=256, max_length=1024,
                 convolutions=((512, 3),) * 20, attention=True, output_attention=False,
                 dropout=0.1, share_embed=False, normalization_constant=0.5, left_pad=False,
                 positional_embeddings=True, prefix=None, params=None):
        super(FConvDecoder, self).__init__(prefix=None, params=None)
        # self.register_buffer('version', torch.Tensor([2]))
        self._dropout = dropout
        self._normalization_constant = normalization_constant
        self._left_pad = left_pad
        self._positional_embeddings = positional_embeddings

        convolutions = extend_conv_spec(convolutions)
        self._convolutions = convolutions
        self._output_attention = output_attention
        self._max_length = max_length
        self._embed_dim = embed_dim
        if isinstance(attention, bool):
            attention = [attention] * len(convolutions)
        if not isinstance(attention, list) or len(attention) != len(convolutions):
            raise ValueError('Attention is expected to be a list of booleans of '
                             'length equal to the number of layers.')
        self._is_attentions = attention
        self._num_attn_layers = sum(attention)

        in_channels = convolutions[0][0]
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.position_weight = self.params.get_constant('const',
                _position_encoding_init(max_length, embed_dim)) if positional_embeddings else None

            self.fc1 = nn.Dense(units=in_channels, flatten=False, in_units=embed_dim, prefix='fc1_')
            self.projections = nn.HybridSequential()
            self.convolutions = nn.HybridSequential()
            self.attentions = nn.HybridSequential()
            self.residuals = []

            layers_in_channels = [in_channels]
            for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
                if residual == 0:
                    residual_dim = out_channels
                else:
                    residual_dim = layers_in_channels[-residual]
                self.projections.add(nn.Dense(out_channels, flatten=False, in_units=in_channels,
                                              prefix='proj%d_' % i)
                                              if residual_dim != out_channels
                                              else nn.HybridLambda('identity', prefix='proj%d_' % i))
                self.convolutions.add(nn.Conv1D(out_channels * 2, kernel_size, in_channels=in_channels,
                                                prefix='conv%d_' % i))
                self.attentions.add(FConvAttentionLayer(out_channels, embed_dim, prefix='attn%d_' % i)
                                                   if attention[i]
                                                   else nn.HybridLambda('identity', prefix='attn%d_' % i))
                self.residuals.append(residual)
                in_channels = out_channels
                layers_in_channels.append(out_channels)
            self.fc2 = nn.Dense(units=out_embed_dim, flatten=False, in_units=in_channels, prefix='fc2_')
    
    def init_state_from_encoder(self, encoder_outputs, encoder_valid_length=None):
        mem_keys, mem_values = encoder_outputs
        decoder_states = [mem_keys, mem_values]
        batch_size, mem_length = mem_values.shape[:-1]
        if encoder_valid_length is not None:
            mem_masks = mx.nd.broadcast_lesser(
                mx.nd.arange(mem_length, ctx=encoder_valid_length.context).reshape((1, -1)),
                encoder_valid_length.reshape((-1, 1)))
        else:
            mem_masks = mx.nd.ones((batch_size, mem_length))
        decoder_states.append(mem_masks)
        self._encoder_valid_length = encoder_valid_length
        return decoder_states
    
    def decode_seq(self, inputs, states):
        output, states, additional_outputs = self.forward(inputs, states)
        return output, states, additional_outputs

    def __call__(self, inputs, states):
        return super(FConvDecoder, self).__call__(inputs, states)
    
    def forward(self, step_input, states):  
        input_shape = step_input.shape
        if len(input_shape) == 2:
            step_input = mx.nd.expand_dims(step_input, axis=1)

        if len(states) == 3:
            batch_size, length, input_dims = step_input.shape

            mem_mask = states[-1]
            # print(mem_mask.shape)
            # print(step_input.shape)
            mem_mask = mx.nd.expand_dims(mem_mask, axis=1)\
                .broadcast_axes(axis=1, size=step_input.shape[1])
            # print(mem_mask.shape)
            states = states[:-1] + [mem_mask]

            incremental_states = []
            in_channels = self._convolutions[0][0]
            for out_channels, kernel_size, _ in self._convolutions:
                incremental_states.append(mx.nd.zeros((batch_size, in_channels, kernel_size-1),
                                                      ctx=step_input.context))
                in_channels = out_channels
            states = [incremental_states] + states

            # Get mem_value length
            # self._src_length = states[2].shape[1]
        
        steps = mx.nd.arange(length, ctx=step_input.context)
        states.append(steps)
        
        step_output, states, step_additional_outputs = super(FConvDecoder, self).forward(step_input,
                                                                                         states)
        states = states[:-1]
        # If it is in testing, only output the last one
        if len(input_shape) == 2:
            step_output = step_output[:, -1, :]
            incremental_states = states[0]
            for i in len(incremental_states):
                incremental_states[i] = incremental_states[i][:, :, 1:]
    
        return step_output, states, step_additional_outputs

    def hybrid_forward(self, F, inputs, states, position_weight=None):
        # if len(states) == 4:
        #     incremental_states, mem_keys, mem_values, steps = states
        #     mem_masks = None
        if len(states) == 5:
            incremental_states, mem_keys, mem_values, mem_masks, steps = states
        else:
            raise ValueError('States is expected to be a list with 5 items')

        # add position embedding
        if self._positional_embeddings:
            position_embed = F.expand_dims(F.Embedding(steps, position_weight,
                                                       self._max_length,
                                                       self._embed_dim), axis=0)
            inputs = F.broadcast_add(inputs, position_embed)

        x = self.dropout_layer(inputs)
        target_embedding = x
        # project to size of convolution
        x = self.fc1(x)  #x.shape (B, T, convolutions[0][0])
        x = self.dropout_layer(x)

        avg_attn_score = []
        residuals = [x]
        for i, (proj, conv, attn, is_attn, res_layer, incr_state) in enumerate(zip(
                                                                        self.projections,
                                                                        self.convolutions,
                                                                        self.attentions,
                                                                        self._is_attentions,
                                                                        self.residuals,
                                                                        incremental_states)):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = proj(x)
            else:
                residual = None
            x = self.dropout_layer(x)
            # B x T x C -> B x C x T
            x = F.swapaxes(x, 1, 2)
            x = F.concat(incr_state, x, dim=-1)
            incremental_states[i] = x
            x = conv(x)
            x = self.dropout_layer(x)
            #x is a Symbol object, so glu function need to know number of channels
            x = glu(x, conv._channels, axis=1)
            # B x C x T -> B x T x C
            x = F.swapaxes(x, 1, 2)
            
            if is_attn:
                x, attn_scores = attn(x, target_embedding, (mem_keys, mem_values), mem_masks)
                if self._output_attention:
                    attn_scores = attn_scores / self._num_attn_layers
                    if len(avg_attn_score) == 0:
                        avg_attn_score.append(attn_scores)
                    else:
                        avg_attn_score[0] = avg_attn_score[0] + attn_scores
            
            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        # project back to size of embedding
        x = self.fc2(x)
        x = self.dropout_layer(x)

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

def extend_conv_spec(convolutions):
    """
    Extends convolutional spec that is a list of tuples of 2 or 3 parameters
    (kernel size, dim size and optionally how many layers behind to look for residual)
    to default the residual propagation param if it is not specified
    """
    extended = []
    for spec in convolutions:
        if len(spec) == 3:
            extended.append(spec)
        elif len(spec) == 2:
            extended.append(spec + (1,))
        else:
            raise Exception('invalid number of parameters in convolution spec ' + str(spec) + '. expected 2 or 3')
    return tuple(extended)

class FConvAttentionLayer(HybridBlock):
    def __init__(self, conv_channels, embed_dim, normalization_constant=0.5,
                 attention_cell='dot', prefix=None, params=None):
        super(FConvAttentionLayer, self).__init__(prefix=prefix, params=params)
        self._normalization_constant = normalization_constant
        # projects from output of convolution to embedding dimension
        self.in_projection = nn.Dense(embed_dim, flatten=False, in_units=conv_channels,
                                      prefix=prefix + 'in_proj_')
        self.attention_layer = _get_attention_cell(attention_cell)
        # projects from embedding dimension to convolution size
        self.out_projection = nn.Dense(conv_channels, flatten=False, in_units=embed_dim,
                                       prefix=prefix + 'out_proj_')
    
    def __call__(self, x, target_embedding, encoder_out, mask):
        return super(FConvAttentionLayer, self).__call__(x, target_embedding, encoder_out, mask)

    def hybrid_forward(self, F, x, target_embedding, encoder_out, mask):
        residual = x

        x = (target_embedding + self.in_projection(x)) * math.sqrt(self._normalization_constant)
        x, attn_scores = self.attention_layer(x, encoder_out[0], encoder_out[1], mask)

        # scale attention output
        s = F.sum(mask, axis=-1, keepdims=True)
        x = F.broadcast_mul(x, s * F.rsqrt(s))

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(self._normalization_constant)
        return x, attn_scores
