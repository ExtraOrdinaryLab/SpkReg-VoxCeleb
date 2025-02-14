import torch
import torch.nn as nn

from .cnn import Conv1d
from .normalization import BatchNorm1d
from ..tdnn_attention import init_weights


class XVectorEncoder(nn.Module):
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
        features: int,
        filters: list,
        kernel_sizes: list,
        dilations: list,
        init_mode: str = 'xavier_uniform',
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        # TDNN layers
        in_channels = features
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