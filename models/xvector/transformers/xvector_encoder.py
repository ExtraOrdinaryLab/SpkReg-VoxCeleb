import torch
import torch.nn as nn


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


class XVectorEncoder(nn.Module):

    def __init__(
        self, 
        features, 
        filters=[512, 512, 512, 512, 1500], 
        kernel_sizes=[5, 3, 3, 1, 1], 
        dilations=[1, 2, 3, 1, 1], 
    ):
        super(XVectorEncoder, self).__init__()
        layers = []
        layers.append(
            TDNNLayer(features, filters[0], kernel_sizes[0], dilations[0])
        )
        for i in range(1, len(filters)):
            layers.append(
                TDNNLayer(filters[i-1], filters[i], kernel_sizes[i], dilations[i])
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x, lengths=None):
        """
        x: tensor of shape (B, C, T)
        """
        out = self.layers(x)
        return out, lengths