import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import NeuralModule
from .tdnn_attention import MaskedSEModule2D


class LFELayer(nn.Module):

    expansion = 1

    def __init__(self, in_planes: int = 1, out_planes: int = 128, stride: int = 1):
        super(LFELayer, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        # Figure 6 from https://arxiv.org/pdf/2401.08342v1
        self.lfe_block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False), 
            nn.ReLU(), 
            nn.BatchNorm2d(out_planes), 
            nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False), 
            nn.ReLU(), 
            nn.BatchNorm2d(out_planes), 
            nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False), 
            nn.ReLU(), 
            nn.BatchNorm2d(out_planes)
        )
        self.se_layer = MaskedSEModule2D(inp_filters=out_planes, se_filters=128, out_filters=out_planes)

        if stride != 1 or stride != (1, 1):
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_planes)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs, length=None):
        x = self.lfe_block(inputs)
        x = self.se_layer(x, length)
        return x + self.shortcut(inputs)


class LFEBlock(nn.Module):

    def __init__(self, in_planes: int = 1, out_planes: int = 128, stride: int = 1, num_blocks: int = 3):
        super(LFEBlock, self).__init__()
        blocks = []
        blocks.append(LFELayer(in_planes, out_planes, stride))

        for _ in range(1, num_blocks):
            blocks.append(LFELayer(in_planes=out_planes, out_planes=out_planes, stride=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, inputs, length=None):
        x = inputs
        for layer in self.blocks:
            x = layer(x, length)
        return x


class LFEEncoder(nn.Module):

    def __init__(self, feat_in: int, filters: list, strides: list, blocks: list):
        super(LFEEncoder, self).__init__()
        lfe_blocks = []
        lfe_blocks.append(
            LFEBlock(feat_in, filters[0], stride=strides[0], num_blocks=blocks[0])
        )

        for idx in range(1, len(blocks)):
            lfe_blocks.append(
                LFEBlock(
                    in_planes=filters[idx-1],
                    out_planes=filters[idx],
                    stride=strides[idx],
                    num_blocks=blocks[idx]
                )
            )

        self.lfe_blocks = nn.Sequential(*lfe_blocks)
        self.flatten = torch.nn.Flatten(1, 2)

    def forward(self, inputs, length=None):
        x = inputs
        for idx, lfe_block in enumerate(self.lfe_blocks):
            x = lfe_block(x, length)
        x = self.flatten(x)
        return x


class TDNNBlock(nn.Module):
    """
    Time-Delay Neural Network (TDNN) Block.
    
    A 1D convolutional block with optional activation and batch normalization.

    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        kernel_size (int, optional): Convolution kernel size. Default: 5.
        stride (int, optional): Convolution stride. Default: 1.
        dilation (int, optional): Dilation rate. Default: 1.
        groups (int, optional): Number of groups in the convolution. Default: 1.
        bias (bool, optional): Whether to use bias in convolution. Default: True.
        activation (nn.Module, optional): Activation function. Default: nn.ReLU.
    """

    def __init__(
        self, 
        input_channels, 
        output_channels, 
        kernel_size=5, 
        stride=1, 
        dilation=1, 
        padding=0, 
        groups=1, 
        bias=True, 
        activation=nn.ReLU
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            input_channels, 
            output_channels,
            kernel_size=kernel_size, 
            stride=stride,
            dilation=dilation, 
            padding=padding, 
            groups=groups, 
            bias=bias
        )

        self.activation = activation() if activation else nn.Identity()
        self.bn = nn.BatchNorm1d(output_channels)

    def forward(self, x):
        """
        Forward pass of TDNNBlock.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, time).
        
        Returns:
            torch.Tensor: Output tensor after convolution, activation, and batch normalization.
        """
        x = self.conv(x)
        x = self.activation(x)
        x = self.bn(x)
        return x


class Bottle2Neck(nn.Module):
    """
    Bottle2Neck block inspired by Res2Net.

    Args:
        in_planes (int): Input channel dimensionality.
        out_planes (int): Output channel dimensionality.
        kernel_size (int, optional): Kernel size for convolution. Default: 3.
        dilation (int, optional): Dilation rate for convolution. Default: 1.
        stride (int, optional): Convolution stride (acts as downsampling). Default: 1.
        scale (int, optional): Number of feature groups in Res2Net-like architecture. Default: 8.
    """

    def __init__(
        self, 
        in_planes, 
        out_planes, 
        kernel_size=3, 
        dilation=1, 
        stride=1, 
        padding=0, 
        scale=8
    ):
        super().__init__()

        if scale < 2:
            raise ValueError("Scale must be at least 2 for Bottle2Neck")

        self.scale = scale
        self.in_planes = in_planes // scale
        self.hidden_planes = out_planes // scale

        self.tdnn_blocks = nn.ModuleList([
            TDNNBlock(
                input_channels=self.in_planes,
                output_channels=self.hidden_planes,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
                padding=padding, 
                activation=nn.ReLU
            )
            for _ in range(scale - 1)
        ])

    def forward(self, x):
        """
        Forward pass of Bottle2Neck.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, time).
        
        Returns:
            torch.Tensor: Concatenated output tensor after processing in feature chunks.
        """
        chunks = torch.chunk(x, self.scale, dim=1)
        outputs = [chunks[0]]  # First chunk is unchanged

        previous = chunks[0]
        for i, chunk in enumerate(chunks[1:], start=1):
            processed = self.tdnn_blocks[i - 1](chunk + previous if i > 1 else chunk)
            outputs.append(processed)
            previous = processed

        return torch.cat(outputs, dim=1)


class GFEEncoder(nn.Module):

    def __init__(self, in_channels: int, filters: list, kernel_sizes: list, scale: int = 8):
        super(GFEEncoder, self).__init__()
        self.gfe_blocks = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, 
                out_channels=filters[0], 
                kernel_size=kernel_sizes[0], 
                padding=(kernel_sizes[0]-1)//2
            ), 
            nn.ReLU(), 
            nn.BatchNorm1d(filters[0]), 
            nn.Conv1d(
                in_channels=filters[0], 
                out_channels=filters[1], 
                kernel_size=kernel_sizes[1], 
                padding=(kernel_sizes[1]-1)//2
            ), 
            nn.ReLU(), 
            nn.BatchNorm1d(filters[1]), 
            Bottle2Neck(
                in_planes=filters[1], 
                out_planes=filters[2], 
                kernel_size=kernel_sizes[2], 
                dilation=1, 
                padding=(kernel_sizes[2]-1)//2, 
                scale=scale
            ), 
            nn.ReLU(), 
            nn.BatchNorm1d(filters[2]), 
            nn.Conv1d(
                in_channels=filters[2], 
                out_channels=filters[3], 
                kernel_size=kernel_sizes[3], 
                padding=(kernel_sizes[3]-1)//2
            ), 
            nn.ReLU(), 
            nn.BatchNorm1d(filters[3]), 
            nn.Conv1d(
                in_channels=filters[3], 
                out_channels=filters[4], 
                kernel_size=kernel_sizes[4], 
                padding=(kernel_sizes[4]-1)//2
            ), 
            nn.ReLU(), 
            nn.BatchNorm1d(filters[4]), 
        )

    def forward(self, x, length=None):
        return self.gfe_blocks(x)