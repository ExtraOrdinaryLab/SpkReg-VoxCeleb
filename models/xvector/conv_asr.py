from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import NeuralModule
from .tdnn_attention import StatsPoolLayer, AttentivePoolLayer, init_weights
from .cnn import Conv1d
from .normalization import BatchNorm1d


class TDNNLayer(nn.Module):

    def __init__(self, in_conv_dim, out_conv_dim, kernel_size, dilation):
        super().__init__()
        self.in_conv_dim = in_conv_dim
        self.out_conv_dim = out_conv_dim
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # for backward compatibility, we keep nn.Linear but call F.conv1d for speed up
        weight = self.kernel.weight.view(self.out_conv_dim, self.kernel_size, self.in_conv_dim).transpose(1, 2)
        hidden_states = nn.functional.conv1d(hidden_states, weight, self.kernel.bias, dilation=self.dilation)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class XVectorEncoder(NeuralModule):
    """
    input:
        feat_in: input feature shape (mel spec feature shape)
        filters: list of filter shapes for SE_TDNN modules
        kernel_sizes: list of kernel shapes for SE_TDNN modules
        dilations: list of dilations for group conv se layer
        scale: scale value to group wider conv channels (deafult:8)

    output:
        outputs : encoded output
        output_length: masked output lengths
    """

    def __init__(
        self,
        feat_in: int,
        filters: list,
        kernel_sizes: list,
        dilations: list,
        init_mode: str = 'xavier_uniform',
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        # TDNN layers
        in_channels = feat_in
        tdnn_blocks = len(filters)
        for block_index in range(tdnn_blocks):
            out_channels = filters[block_index]
            self.blocks.extend(
                [
                    Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_sizes[block_index],
                        dilation=dilations[block_index],
                    ),
                    torch.nn.LeakyReLU(),
                    BatchNorm1d(input_size=out_channels),
                ]
            )
            in_channels = filters[block_index]

        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor = None):
        """
        audio_signal: tensor shape of (B, D, T)
        output: tensor shape of (B, D, T)
        """
        x = audio_signal.transpose(1, 2)
        for layer in self.blocks:
            x = layer(x)
        output = x.transpose(1, 2)
        return output, length


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

    def forward(self, encoder_output, length: torch.Tensor = None):
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