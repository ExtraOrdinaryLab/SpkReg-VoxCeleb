import torch
import torch.nn as nn
import torch.nn.functional as F


class Res2Conv1dReluBn(nn.Module):
    
    def __init__(
        self,
        channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        scale=4
    ):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias=bias
                )
            )
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            # Order: conv -> relu -> bn
            if i >= 1:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out


class Conv1dReluBn(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SEConnect(nn.Module):

    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x: torch.Tensor):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


class SERes2Block(nn.Module):

    def __init__(
        self, channels, kernel_size, stride, padding, dilation, scale, se_channels=128
    ):
        super().__init__()
        self.se_res2block = nn.Sequential(
            Conv1dReluBn(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            Res2Conv1dReluBn(
                channels,
                kernel_size,
                stride,
                padding,
                dilation,
                scale=scale
            ),
            Conv1dReluBn(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0
            ), 
            SEConnect(
                channels, 
                se_channels
            )
        )

    def forward(self, x):
        return x + self.se_res2block(x)


class EcapaTdnnEncoder(nn.Module):

    def __init__(
        self,
        features, 
        filters=[512, 512, 512, 512, 1536],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        res2net_scale=8,
        se_channels=128,
    ):
        super().__init__()

        self.layer1 = Conv1dReluBn(
            features,
            filters[0],
            kernel_size=kernel_sizes[0],
            padding=2, 
            dilation=dilations[0]
        )
        self.layer2 = SERes2Block(
            filters[1],
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=2,
            dilation=dilations[1],
            scale=res2net_scale, 
            se_channels=se_channels
        )
        self.layer3 = SERes2Block(
            filters[2],
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=3,
            dilation=dilations[2],
            scale=res2net_scale, 
            se_channels=se_channels
        )
        self.layer4 = SERes2Block(
            filters[3],
            kernel_size=kernel_sizes[3],
            stride=1,
            padding=4,
            dilation=dilations[3],
            scale=res2net_scale, 
            se_channels=se_channels
        )

        cat_channels = filters[3] + filters[2] + filters[1]
        out_channels = filters[4]
        self.conv = nn.Conv1d(
            cat_channels, 
            out_channels, 
            kernel_size=1, 
            dilation=dilations[4]
        )

    def get_frame_level_feat(self, x):
        """
        x: tensor of shape (B, C, T)
        """
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = torch.cat([out2, out3, out4], dim=1)
        out = self.conv(out)

        return out

    def forward(self, x, lengths=None):
        """
        x: tensor of shape (B, C, T)
        """
        out = self.get_frame_level_feat(x)
        out = F.relu(out)
        return out, lengths