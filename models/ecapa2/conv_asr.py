from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import NeuralModule
from .tdnn_attention import (
    StatsPoolLayer, 
    AttentivePoolLayer, 
    ChannelDependentAttentiveStatisticsPoolLayer, 
    TdnnModule, 
    TdnnSeModule, 
    TdnnSeRes2NetModule, 
    init_weights
)
from .ecapa2 import LFEEncoder, GFEEncoder


class Ecapa2Encoder(NeuralModule):

    def __init__(
        self, 
        feat_in: int = 256, 
        lfe_filters: list = [164, 164, 164, 192, 192], 
        lfe_strides: list = [(1, 1), (2, 1), (2, 1), (2, 1), (2, 1)], 
        lfe_blocks: list = [3, 4, 4, 4, 5], 
        gfe_filters: list = [1024, 1024, 1024, 1024, 1536], 
        gfe_kernel_sizes: list = [1, 1, 3, 1, 1], 
        res2net_scale: int = 8, 
        init_mode: str = 'xavier_uniform',
    ):
        super().__init__()
        self.feat_in = feat_in
        self.filters = lfe_filters
        self.strides = lfe_strides

        self.lfe_module = LFEEncoder(
            feat_in=1, 
            filters=lfe_filters, 
            strides=lfe_strides, 
            blocks=lfe_blocks, 
        )
        gfe_height, gfe_width = self.get_gfe_input_shape()
        self.gfe_module = GFEEncoder(
            in_channels=gfe_height*gfe_width, 
            filters=gfe_filters, 
            kernel_sizes=gfe_kernel_sizes, 
            scale=res2net_scale
        )

        self.apply(lambda x: init_weights(x, mode=init_mode))

    def get_gfe_input_shape(self):
        height = self.feat_in
        for stride in self.strides:
            if isinstance(stride, tuple) or isinstance(stride, list):
                stride = stride[0]
            height = height // stride
        width = self.filters[-1]
        return height, width

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor = None):
        # Figure 5 from https://arxiv.org/pdf/2401.08342v1
        x = audio_signal
        x = x.unsqueeze(dim=1) # (B, 1, 256, T)
        x = self.lfe_module(x, length) # (B, 16, 192, T)
        x = self.gfe_module(x, length) # (B, 1536, T)
        return x, length


class SpeakerDecoder(NeuralModule):
    """
    Speaker Decoder creates the final neural layers that maps from the outputs
    of Jasper Encoder to the embedding layer followed by speaker based softmax loss.

    Args:
        feat_in (int): Number of channels being input to this module
        num_classes (int): Number of unique speakers in dataset
        emb_sizes (list) : shapes of intermediate embedding layers (we consider speaker embbeddings
            from 1st of this layers). Defaults to [1024,1024]
        pool_mode (str) : Pooling strategy type. options are 'xvector','tap', 'attention'
            Defaults to 'xvector (mean and variance)'
            tap (temporal average pooling: just mean)
            attention (attention based pooling)
        init_mode (str): Describes how neural network parameters are
            initialized. Options are ['xavier_uniform', 'xavier_normal',
            'kaiming_uniform','kaiming_normal'].
            Defaults to "xavier_uniform".
    """

    def __init__(
        self,
        feat_in: int,
        num_classes: int,
        emb_sizes: Optional[Union[int, list]] = 256,
        pool_mode: str = 'xvector',
        angular: bool = False,
        attention_channels: int = 128,
        init_mode: str = "xavier_uniform",
    ):
        super().__init__()
        self.angular = angular
        self.emb_id = 2
        bias = False if self.angular else True
        emb_sizes = [emb_sizes] if type(emb_sizes) is int else emb_sizes

        self._num_classes = num_classes
        self.pool_mode = pool_mode.lower()
        if self.pool_mode == 'xvector' or self.pool_mode == 'tap':
            self._pooling = StatsPoolLayer(feat_in=feat_in, pool_mode=self.pool_mode)
            affine_type = 'linear'
        elif self.pool_mode == 'attention':
            self._pooling = AttentivePoolLayer(inp_filters=feat_in, attention_channels=attention_channels)
            affine_type = 'conv'
        elif self.pool_mode == 'ecapa2':
            self._pooling = ChannelDependentAttentiveStatisticsPoolLayer(
                inp_filters=feat_in, attention_channels=attention_channels
            )
            affine_type = 'conv'

        shapes = [self._pooling.feat_in]
        for size in emb_sizes:
            shapes.append(int(size))

        emb_layers = []
        for shape_in, shape_out in zip(shapes[:-1], shapes[1:]):
            layer = self.affine_layer(shape_in, shape_out, learn_mean=False, affine_type=affine_type)
            emb_layers.append(layer)

        self.emb_layers = nn.ModuleList(emb_layers)

        self.final = nn.Linear(shapes[-1], self._num_classes, bias=bias)

        self.apply(lambda x: init_weights(x, mode=init_mode))

    def affine_layer(
        self,
        inp_shape,
        out_shape,
        learn_mean=True,
        affine_type='conv',
    ):
        if affine_type == 'conv':
            layer = nn.Sequential(
                nn.BatchNorm1d(inp_shape, affine=True, track_running_stats=True),
                nn.Conv1d(inp_shape, out_shape, kernel_size=1),
            )

        else:
            layer = nn.Sequential(
                nn.Linear(inp_shape, out_shape),
                nn.BatchNorm1d(out_shape, affine=learn_mean, track_running_stats=True),
                nn.ReLU(),
            )

        return layer

    def forward(self, encoder_output, length=None):
        pool = self._pooling(encoder_output, length)
        embs = []

        for layer in self.emb_layers:
            pool, emb = layer(pool), layer[: self.emb_id](pool)
            embs.append(emb)

        pool = pool.squeeze(-1)
        if self.angular:
            for W in self.final.parameters():
                W = F.normalize(W, p=2, dim=1)
            pool = F.normalize(pool, p=2, dim=1)

        out = self.final(pool)

        return out, embs[-1].squeeze(-1)