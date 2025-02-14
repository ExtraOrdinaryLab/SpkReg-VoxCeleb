"""
WeSpeaker and SpeakerLab use the same implementation for ECAPA-TDNN.

https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/models/tdnn.py
https://github.com/modelscope/3D-Speaker/blob/main/speakerlab/models/xvector/TDNN.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TdnnLayer(nn.Module):

    def __init__(self, in_dim, out_dim, context_size, dilation=1, padding=0):
        super(TdnnLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.context_size = context_size
        self.dilation = dilation
        self.padding = padding
        self.conv_1d = nn.Conv1d(
            self.in_dim,
            self.out_dim,
            self.context_size,
            dilation=self.dilation,
            padding=self.padding
        )
        self.bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        out = self.conv_1d(x)
        out = F.relu(out)
        out = self.bn(out)
        return out


class XVectorEncoder(nn.Module):

    def __init__(
        self,
        features, 
        filters=[512, 512, 512, 512, 1500], 
        kernel_sizes=[5, 3, 3, 1, 1], 
        dilations=[1, 2, 3, 1, 1], 
    ):
        super(XVectorEncoder, self).__init__()
        self.frame_1 = TdnnLayer(
            features, filters[0], context_size=kernel_sizes[0], dilation=dilations[0]
        )
        self.frame_2 = TdnnLayer(
            filters[0], filters[1], context_size=kernel_sizes[1], dilation=dilations[1]
        )
        self.frame_3 = TdnnLayer(
            filters[1], filters[2], context_size=kernel_sizes[2], dilation=dilations[2]
        )
        self.frame_4 = TdnnLayer(
            filters[2], filters[3], context_size=kernel_sizes[3], dilation=dilations[3]
        )
        self.frame_5 = TdnnLayer(
            filters[3], filters[4], context_size=kernel_sizes[4], dilation=dilations[4]
        )

    def get_frame_level_feat(self, x):
        out = self.frame_1(x)
        out = self.frame_2(out)
        out = self.frame_3(out)
        out = self.frame_4(out)
        out = self.frame_5(out)
        return out

    def forward(self, x, lengths=None):
        out = self.get_frame_level_feat(x)
        return out, lengths