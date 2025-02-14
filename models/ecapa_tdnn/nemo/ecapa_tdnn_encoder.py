import torch
import torch.nn as nn

from ..tdnn_attention import (
    TdnnModule, 
    TdnnSeModule, 
    TdnnSeRes2NetModule, 
    init_weights
)


class EcapaTdnnEncoder(nn.Module):
    """
    Modified ECAPA Encoder layer without Res2Net module for faster training and inference which achieves
    better numbers on speaker diarization tasks
    Reference: ECAPA-TDNN Embeddings for Speaker Diarization (https://arxiv.org/pdf/2104.01466.pdf)

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
        features: int,
        filters: list,
        kernel_sizes: list,
        dilations: list,
        se_channels: int = 128, 
        res2net: bool = True, 
        res2net_scale: int = 8, 
        init_mode: str = 'xavier_uniform',
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            TdnnModule(features, filters[0], kernel_size=kernel_sizes[0], dilation=dilations[0])
        )

        for i in range(len(filters) - 2):
            if res2net:
                self.layers.append(
                    TdnnSeRes2NetModule(
                        filters[i],
                        filters[i + 1],
                        group_scale=1,
                        se_channels=se_channels,
                        kernel_size=kernel_sizes[i + 1],
                        dilation=dilations[i + 1],
                        res2net_scale=res2net_scale, 
                    )
                )
            else:
                self.layers.append(
                    TdnnSeModule(
                        filters[i],
                        filters[i + 1],
                        group_scale=1,
                        se_channels=128,
                        kernel_size=kernel_sizes[i + 1],
                        dilation=dilations[i + 1],
                    )
                )
        self.feature_agg = TdnnModule(filters[-1], filters[-1], kernel_sizes[-1], dilations[-1])
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, audio_signal, length=None):
        x = audio_signal
        outputs = []

        for layer in self.layers:
            x = layer(x, length=length)
            outputs.append(x)

        x = torch.cat(outputs[1:], dim=1)
        x = self.feature_agg(x)
        return x, length