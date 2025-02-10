import math
from typing import List, Optional

from numpy import inf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_correct_fan


class StatsPoolLayer(nn.Module):
    """Statistics and time average pooling (TAP) layer

    This computes mean and, optionally, standard deviation statistics across the time dimension.

    Args:
        feat_in: Input features with shape [B, D, T]
        pool_mode: Type of pool mode. Supported modes are 'xvector' (mean and standard deviation) and 'tap' (time
            average pooling, i.e., mean)
        eps: Epsilon, minimum value before taking the square root, when using 'xvector' mode.
        unbiased: Whether to use the biased estimator for the standard deviation when using 'xvector' mode. The default
            for torch.Tensor.std() is True.

    Returns:
        Pooled statistics with shape [B, D].

    Raises:
        ValueError if an unsupported pooling mode is specified.
    """

    def __init__(self, feat_in: int, pool_mode: str = 'xvector', eps: float = 1e-10, unbiased: bool = True):
        super().__init__()
        supported_modes = {"xvector", "tap"}
        if pool_mode not in supported_modes:
            raise ValueError(f"Pool mode must be one of {supported_modes}; got '{pool_mode}'")
        self.pool_mode = pool_mode
        self.feat_in = feat_in
        self.eps = eps
        self.unbiased = unbiased
        if self.pool_mode == 'xvector':
            # Mean + std
            self.feat_in *= 2

    def forward(self, encoder_output, length=None):
        if length is None:
            mean = encoder_output.mean(dim=-1)  # Time Axis
            if self.pool_mode == 'xvector':
                correction = 1 if self.unbiased else 0
                std = encoder_output.std(dim=-1, correction=correction).clamp(min=self.eps)
                pooled = torch.cat([mean, std], dim=-1)
            else:
                pooled = mean
        else:
            mask = make_seq_mask_like(like=encoder_output, lengths=length, valid_ones=False)
            encoder_output = encoder_output.masked_fill(mask, 0.0)
            # [B, D, T] -> [B, D]
            means = encoder_output.mean(dim=-1)
            # Re-scale to get padded means
            means = means * (encoder_output.shape[-1] / length).unsqueeze(-1)
            if self.pool_mode == "xvector":
                correction = 1 if self.unbiased else 0
                stds = (
                    encoder_output.sub(means.unsqueeze(-1))
                    .masked_fill(mask, 0.0)
                    .pow(2.0)
                    .sum(-1)  # [B, D, T] -> [B, D]
                    .div(length.view(-1, 1).sub(correction))
                    .clamp(min=self.eps)
                    .sqrt()
                )
                pooled = torch.cat((means, stds), dim=-1)
            else:
                pooled = means
        return pooled


class AttentivePoolLayer(nn.Module):
    """
    Attention pooling layer for pooling speaker embeddings
    Reference: ECAPA-TDNN Embeddings for Speaker Diarization (https://arxiv.org/pdf/2104.01466.pdf)
    inputs:
        inp_filters: input feature channel length from encoder
        attention_channels: intermediate attention channel size
        kernel_size: kernel_size for TDNN and attention conv1d layers (default: 1)
        dilation: dilation size for TDNN and attention conv1d layers  (default: 1)
    """

    def __init__(
        self,
        inp_filters: int,
        attention_channels: int = 128,
        kernel_size: int = 1,
        dilation: int = 1,
        eps: float = 1e-10,
    ):
        super().__init__()

        self.feat_in = 2 * inp_filters

        self.attention_layer = nn.Sequential(
            TdnnModule(inp_filters * 3, attention_channels, kernel_size=kernel_size, dilation=dilation),
            nn.Tanh(),
            nn.Conv1d(
                in_channels=attention_channels,
                out_channels=inp_filters,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
        )
        self.eps = eps

    def forward(self, x, length=None):
        max_len = x.size(2)

        if length is None:
            length = torch.ones(x.shape[0], device=x.device)

        mask, num_values = lens_to_mask(length, max_len=max_len, device=x.device)

        # encoder statistics
        mean, std = get_statistics_with_mask(x, mask / num_values)
        mean = mean.unsqueeze(2).repeat(1, 1, max_len)
        std = std.unsqueeze(2).repeat(1, 1, max_len)
        attn = torch.cat([x, mean, std], dim=1)

        # attention statistics
        attn = self.attention_layer(attn)  # attention pass
        attn = attn.masked_fill(mask == 0, -inf)
        alpha = F.softmax(attn, dim=2)  # attention values, α
        mu, sg = get_statistics_with_mask(x, alpha)  # µ and ∑

        # gather
        return torch.cat((mu, sg), dim=1).unsqueeze(2)


class TdnnModule(nn.Module):
    """
    Time Delayed Neural Module (TDNN) - 1D
    input:
        inp_filters: input filter channels for conv layer
        out_filters: output filter channels for conv layer
        kernel_size: kernel weight size for conv layer
        dilation: dilation for conv layer
        stride: stride for conv layer
        padding: padding for conv layer (default None: chooses padding value such that input and output feature shape matches)
    output:
        tdnn layer output
    """

    def __init__(
        self,
        inp_filters: int,
        out_filters: int,
        kernel_size: int = 1,
        dilation: int = 1,
        stride: int = 1, 
        groups: int = 1,
        padding: int = None,
    ):
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size, stride=stride, dilation=dilation)

        self.conv_layer = nn.Conv1d(
            in_channels=inp_filters,
            out_channels=out_filters,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups, 
            padding=padding,
        )

        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_filters)

    def forward(self, x, length=None):
        x = self.conv_layer(x)
        x = self.activation(x)
        return self.bn(x)


class MaskedSEModule(nn.Module):
    """
    Squeeze and Excite module implementation with conv1d layers
    input:
        inp_filters: input filter channel size
        se_filters: intermediate squeeze and excite channel output and input size
        out_filters: output filter channel size
        kernel_size: kernel_size for both conv1d layers
        dilation: dilation size for both conv1d layers

    output:
        squeeze and excite layer output
    """

    def __init__(self, inp_filters: int, se_filters: int, out_filters: int, kernel_size: int = 1, dilation: int = 1):
        super().__init__()
        self.se_layer = nn.Sequential(
            nn.Conv1d(
                inp_filters,
                se_filters,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(se_filters),
            nn.Conv1d(
                se_filters,
                out_filters,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
            nn.Sigmoid(),
        )

    def forward(self, input, length=None):
        if length is None:
            x = torch.mean(input, dim=2, keep_dim=True)
        else:
            max_len = input.size(2)
            mask, num_values = lens_to_mask(length, max_len=max_len, device=input.device)
            x = torch.sum((input * mask), dim=2, keepdim=True) / (num_values)

        out = self.se_layer(x)
        return out * input


class TdnnSeModule(nn.Module):
    """
    Modified building SE_TDNN group module block from ECAPA implementation for faster training and inference
    Reference: ECAPA-TDNN Embeddings for Speaker Diarization (https://arxiv.org/pdf/2104.01466.pdf)
    inputs:
        inp_filters: input filter channel size
        out_filters: output filter channel size
        group_scale: scale value to group wider conv channels (deafult:8)
        se_channels: squeeze and excite output channel size (deafult: 1024/8= 128)
        kernel_size: kernel_size for group conv1d layers (default: 1)
        dilation: dilation size for group conv1d layers  (default: 1)
    """

    def __init__(
        self,
        inp_filters: int,
        out_filters: int,
        group_scale: int = 8,
        se_channels: int = 128,
        kernel_size: int = 1,
        dilation: int = 1,
        init_mode: str = 'xavier_uniform',
    ):
        super().__init__()
        self.out_filters = out_filters
        padding_val = get_same_padding(kernel_size=kernel_size, dilation=dilation, stride=1)

        group_conv = nn.Conv1d(
            out_filters,
            out_filters,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding_val,
            groups=group_scale,
        )
        self.group_tdnn_block = nn.Sequential(
            TdnnModule(inp_filters, out_filters, kernel_size=1, dilation=1),
            group_conv,
            nn.ReLU(),
            nn.BatchNorm1d(out_filters),
            TdnnModule(out_filters, out_filters, kernel_size=1, dilation=1),
        )

        self.se_layer = MaskedSEModule(out_filters, se_channels, out_filters)

        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, inputs, length=None):
        x = self.group_tdnn_block(inputs)
        x = self.se_layer(x, length)
        return x + inputs


class Res2NetBlock(nn.Module):
    """
    Res2Net module that splits input channels into groups and processes them separately before merging.
    This allows multi-scale feature extraction.
    """
    def __init__(self, in_channels, out_channels, scale=4, kernel_size=1, dilation=1):
        super().__init__()
        assert in_channels % scale == 0, "in_channels must be divisible by scale"
        
        self.scale = scale
        self.width = in_channels // scale  # Number of channels per group
        
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size=kernel_size, dilation=dilation, padding=dilation, bias=False)
            for _ in range(scale - 1)
        ])
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        x: [B, C, T]
        """
        splits = torch.split(x, self.width, dim=1)
        outputs = [splits[0]]  # First part remains unchanged

        for i in range(1, self.scale):
            conv_out = self.convs[i - 1](splits[i])  # Apply convolution on each group
            outputs.append(conv_out + outputs[i - 1])  # Hierarchical aggregation

        out = torch.cat(outputs, dim=1)  # Merge groups
        return self.activation(self.bn(out))


class TdnnSeRes2NetModule(nn.Module):
    """
    SE-TDNN module with Res2Net for ECAPA-TDNN.
    """
    def __init__(
        self,
        inp_filters: int,
        out_filters: int,
        group_scale: int = 1,
        se_channels: int = 128,
        kernel_size: int = 1,
        dilation: int = 1,
        res2net_scale: int = 8,  # New Res2Net parameter
    ):
        super().__init__()

        # First TDNN layer
        self.tdnn1 = TdnnModule(inp_filters, out_filters, kernel_size=1, dilation=1, groups=group_scale)

        # Res2Net block replaces grouped TDNN
        self.res2net = Res2NetBlock(out_filters, out_filters, scale=res2net_scale, kernel_size=kernel_size, dilation=dilation)

        # Squeeze-and-Excite module
        self.se_layer = MaskedSEModule(out_filters, se_channels, out_filters)

    def forward(self, x, length=None):
        residual = x
        x = self.tdnn1(x)
        x = self.res2net(x)  # Apply Res2Net block
        x = self.se_layer(x, length)
        return x + residual  # Residual connection


class MaskedConv1d(nn.Module):
    
    __constants__ = ["use_conv_mask", "real_out_channels", "heads"]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        heads=-1,
        bias=False,
        use_mask=True,
        quantize=False,
    ):
        super(MaskedConv1d, self).__init__()

        if not (heads == -1 or groups == in_channels):
            raise ValueError("Only use heads for depthwise convolutions")

        self.real_out_channels = out_channels
        if heads != -1:
            in_channels = heads
            out_channels = heads
            groups = heads

        # preserve original padding
        self._padding = padding

        # if padding is a tuple/list, it is considered as asymmetric padding
        if type(padding) in (tuple, list):
            self.pad_layer = nn.ConstantPad1d(padding, value=0.0)
            # reset padding for conv since pad_layer will handle this
            padding = 0
        else:
            self.pad_layer = None

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.use_mask = use_mask
        self.heads = heads

        # Calculations for "same" padding cache
        self.same_padding = (self.conv.stride[0] == 1) and (
            2 * self.conv.padding[0] == self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
        )
        if self.pad_layer is None:
            self.same_padding_asymmetric = False
        else:
            self.same_padding_asymmetric = (self.conv.stride[0] == 1) and (
                sum(self._padding) == self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
            )

        # `self.lens` caches consecutive integers from 0 to `self.max_len` that are used to compute the mask for a
        # batch. Recomputed to bigger size as needed. Stored on a device of the latest batch lens.
        if self.use_mask:
            self.max_len = torch.tensor(0)
            self.lens = torch.tensor(0)

    def get_seq_len(self, lens):
        if self.same_padding or self.same_padding_asymmetric:
            return lens

        if self.pad_layer is None:
            return (
                torch.div(
                    lens + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1,
                    self.conv.stride[0],
                    rounding_mode='trunc',
                )
                + 1
            )
        else:
            return (
                torch.div(
                    lens + sum(self._padding) - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1,
                    self.conv.stride[0],
                    rounding_mode='trunc',
                )
                + 1
            )

    def forward(self, x, lens):
        if self.use_mask:
            # Generally will be called by ConvASREncoder, but kept as single gpu backup.
            if x.size(2) > self.max_len:
                self.update_masked_length(x.size(2), device=lens.device)
            x = self.mask_input(x, lens)

        # Update lengths
        lens = self.get_seq_len(lens)

        # asymmtric pad if necessary
        if self.pad_layer is not None:
            x = self.pad_layer(x)

        sh = x.shape
        if self.heads != -1:
            x = x.view(-1, self.heads, sh[-1])

        out = self.conv(x)

        if self.heads != -1:
            out = out.view(sh[0], self.real_out_channels, -1)

        return out, lens

    def update_masked_length(self, max_len, seq_range=None, device=None):
        if seq_range is None:
            self.lens, self.max_len = _masked_conv_init_lens(self.lens, max_len, self.max_len)
            self.lens = self.lens.to(device)
        else:
            self.lens = seq_range
            self.max_len = torch.tensor(max_len)

    def mask_input(self, x, lens):
        max_len = x.size(2)
        mask = self.lens[:max_len].unsqueeze(0).to(lens.device) < lens.unsqueeze(1)
        x = x * mask.unsqueeze(1).to(device=x.device)
        return x


@torch.jit.script
def _masked_conv_init_lens(lens: torch.Tensor, current_maxlen: int, original_maxlen: torch.Tensor):
    if current_maxlen > original_maxlen:
        new_lens = torch.arange(current_maxlen)
        new_max_lens = torch.tensor(current_maxlen)
    else:
        new_lens = lens
        new_max_lens = original_maxlen
    return new_lens, new_max_lens


def get_same_padding(kernel_size, stride, dilation) -> int:
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (dilation * (kernel_size - 1)) // 2


def lens_to_mask(lens: List[int], max_len: int, device: str = None):
    """
    outputs masking labels for list of lengths of audio features, with max length of any
    mask as max_len
    input:
        lens: list of lens
        max_len: max length of any audio feature
    output:
        mask: masked labels
        num_values: sum of mask values for each feature (useful for computing statistics later)
    """
    lens_mat = torch.arange(max_len).to(device)
    mask = lens_mat[:max_len].unsqueeze(0) < lens.unsqueeze(1)
    mask = mask.unsqueeze(1)
    num_values = torch.sum(mask, dim=2, keepdim=True)
    return mask, num_values


def get_statistics_with_mask(x: torch.Tensor, m: torch.Tensor, dim: int = 2, eps: float = 1e-10):
    """
    compute mean and standard deviation of input(x) provided with its masking labels (m)
    input:
        x: feature input
        m: averaged mask labels
    output:
        mean: mean of input features
        std: stadard deviation of input features
    """
    mean = torch.sum((m * x), dim=dim)
    std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
    return mean, std


@torch.jit.script_if_tracing
def make_seq_mask_like(
    like: torch.Tensor, lengths: torch.Tensor, valid_ones: bool = True, time_dim: int = -1
) -> torch.Tensor:
    mask = torch.arange(like.shape[time_dim], device=like.device).repeat(lengths.shape[0], 1).lt(lengths.unsqueeze(-1))
    # Match number of dims in `like` tensor
    for _ in range(like.dim() - mask.dim()):
        mask = mask.unsqueeze(1)
    # If time dim != -1, transpose to proper dim.
    if time_dim != -1:
        mask = mask.transpose(time_dim, -1)
    if not valid_ones:
        mask = ~mask
    return mask


def init_weights(m, mode: Optional[str] = 'xavier_uniform'):
    if isinstance(m, MaskedConv1d):
        init_weights(m.conv, mode)
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        if mode is not None:
            if mode == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif mode == 'xavier_normal':
                nn.init.xavier_normal_(m.weight, gain=1.0)
            elif mode == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif mode == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif mode == 'tds_uniform':
                tds_uniform_(m.weight)
            elif mode == 'tds_normal':
                tds_normal_(m.weight)
            else:
                raise ValueError("Unknown Initialization mode: {0}".format(mode))
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def tds_uniform_(tensor, mode='fan_in'):
    """
    Uniform Initialization from the paper [Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2460.pdf)
    Normalized to -

    .. math::
        \\text{bound} = \\text{2} \\times \\sqrt{\\frac{1}{\\text{fan\\_mode}}}

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = 2.0  # sqrt(4.0) = 2
    std = gain / math.sqrt(fan)  # sqrt(4.0 / fan_in)
    bound = std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def tds_normal_(tensor, mode='fan_in'):
    """
    Normal Initialization from the paper [Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2460.pdf)
    Normalized to -

    .. math::
        \\text{bound} = \\text{2} \\times \\sqrt{\\frac{1}{\\text{fan\\_mode}}}

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = 2.0
    std = gain / math.sqrt(fan)  # sqrt(4.0 / fan_in)
    bound = std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.normal_(0.0, bound)