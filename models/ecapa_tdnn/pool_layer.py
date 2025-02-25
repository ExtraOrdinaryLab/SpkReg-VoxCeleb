import math
from typing import List

from numpy import inf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.init import _calculate_correct_fan

from .tdnn_attention import TdnnModule


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


class MpnCovPoolLayer(nn.Module):

    def __init__(self, use_sqrt=True, use_vector=True, input_dim=2048, dimension_reduction=None):
        super(MpnCovPoolLayer, self).__init__()
        self.use_sqrt = use_sqrt
        self.use_vector = use_vector
        self.dimension_reduction = dimension_reduction

        # Optionally reduce feature dimensionality via a 1x1 convolution block.
        if self.dimension_reduction is not None:
            self.conv_dim_reduction = nn.Sequential(
                nn.Conv2d(input_dim, self.dimension_reduction, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.dimension_reduction),
                nn.ReLU(inplace=True)
            )

        output_dim = self.dimension_reduction if self.dimension_reduction is not None else input_dim
        # Compute output dimension: if vectorized, use the number of upper triangular entries.
        if self.use_vector:
            self.output_dim = output_dim * (output_dim + 1) // 2
        else:
            self.output_dim = output_dim * output_dim

        # The output dimension
        self.feat_in = self.output_dim

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights for convolution and batch normalization layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _covariance_pooling(self, input_tensor: torch.Tensor):
        """
        Forward pass for covariance pooling.

        Args:
            input_tensor (torch.Tensor): Input feature map with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Covariance matrix with shape (batch_size, channels, channels).
        """
        batch_size, channels, height, width = input_tensor.shape
        num_elements = height * width

        # Reshape input tensor to (batch_size, channels, num_elements)
        x_reshaped = input_tensor.view(batch_size, channels, num_elements)
        # Compute the mean for each channel (per sample)
        mean = x_reshaped.mean(dim=2, keepdim=True)
        # Center the data
        x_centered = x_reshaped - mean
        # Compute covariance: (1/(N-1)) * X_centered * X_centered^T
        covariance = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (num_elements - 1)
        assert covariance.shape == (batch_size, channels, channels)
        return covariance

    def _matrix_square_root(self, input_tensor: torch.Tensor):
        """
        Forward pass to compute the matrix square root.

        Args:
            input_tensor (torch.Tensor): Input covariance matrix with shape (batch_size, channels, channels).
            iter_num (int): Number of iterations for the Newton-Schulz method.

        Returns:
            torch.Tensor: Matrix square root of the input.
        """
        # Compute the eigenvalue decomposition. For symmetric matrices, use torch.linalg.eigh.
        eigvals, eigvecs = torch.linalg.eigh(input_tensor)
        # Clamp eigenvalues for numerical stability (avoid negative values due to numerical issues)
        eps = 1e-6
        eigvals = torch.clamp(eigvals, min=eps)
        # Compute the square root of the eigenvalues
        sqrt_eigvals = torch.sqrt(eigvals)
        # Reconstruct the square root matrix: U * diag(sqrt_eigvals) * U^T
        sqrt_diag = torch.diag_embed(sqrt_eigvals)
        sqrt_matrix = eigvecs @ sqrt_diag @ eigvecs.transpose(-2, -1)
        assert sqrt_matrix.shape == input_tensor.shape
        return sqrt_matrix

    def _upper_triangular_vector(self, input_tensor: torch.Tensor):
        """
        Forward pass to extract the upper triangular (including diagonal) elements and vectorize them.

        Args:
            input_tensor (torch.Tensor): Input matrix of shape (batch_size, channels, channels).

        Returns:
            torch.Tensor: Vectorized upper triangular part with shape (batch_size, channels*(channels+1)//2).
        """
        batch_size, channels, _ = input_tensor.shape
        # Get the row and column indices for the upper triangular part
        row_idx, col_idx = torch.triu_indices(channels, channels, device=input_tensor.device)
        # Use advanced indexing to extract the upper triangular values for each sample
        upper_tri_vec = input_tensor[:, row_idx, col_idx]
        assert upper_tri_vec.shape == (batch_size, channels*(channels+1)//2)
        return upper_tri_vec

    def forward(self, x: torch.Tensor, length=None):
        """
        Forward pass of the MPNCOV module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying covariance pooling, optional square root normalization,
                          and optional vectorization.
        """
        if x.ndim == 3: # Audio (B, D, T)
            x = x.unsqueeze(2)
        if self.dimension_reduction is not None:
            x = self.conv_dim_reduction(x)
        x = self._covariance_pooling(x)
        if self.use_sqrt:
            x = self._matrix_square_root(x)
        if self.use_vector:
            x = self._upper_triangular_vector(x)
        return x


class iSqrtCovPoolLayer(nn.Module):
    """
    Matrix Power Normalized Covariance Pooling (MPNCOV) module.

    This module implements the fast MPN-COV method (iSQRT-COV):
    https://arxiv.org/abs/1712.01034

    Args:
        iter_num (int): Number of iterations for the Newton-Schulz method.
        use_sqrt (bool): Whether to perform matrix square root normalization.
        use_vector (bool): Whether to output a vector (upper triangular part) rather than a full matrix.
        input_dim (int): Number of input feature channels.
        dimension_reduction (int or None): If specified, reduces the input channel dimension using a 1x1 convolution.
    """

    def __init__(self, iter_num=3, use_sqrt=True, use_vector=True, input_dim=2048, dimension_reduction=None):
        super(iSqrtCovPoolLayer, self).__init__()
        self.iter_num = iter_num
        self.use_sqrt = use_sqrt
        self.use_vector = use_vector
        self.dimension_reduction = dimension_reduction

        # Optionally reduce feature dimensionality via a 1x1 convolution block.
        if self.dimension_reduction is not None:
            self.conv_dim_reduction = nn.Sequential(
                nn.Conv2d(input_dim, self.dimension_reduction, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.dimension_reduction),
                nn.ReLU(inplace=True)
            )

        output_dim = self.dimension_reduction if self.dimension_reduction is not None else input_dim
        # Compute output dimension: if vectorized, use the number of upper triangular entries.
        if self.use_vector:
            self.output_dim = output_dim * (output_dim + 1) // 2
        else:
            self.output_dim = output_dim * output_dim

        # The output dimension
        self.feat_in = self.output_dim

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights for convolution and batch normalization layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _covariance_pooling(self, x):
        """
        Perform covariance pooling on the input tensor.
        """
        return CovPoolingFunction.apply(x)

    def _matrix_square_root(self, x):
        """
        Compute the matrix square root using the Newton-Schulz iteration.
        """
        return MatrixSquareRootFunction.apply(x, self.iter_num)

    def _upper_triangular_vector(self, x):
        """
        Extract and vectorize the upper triangular part (including diagonal) of the matrix.
        """
        return UpperTriangularVectorFunction.apply(x)

    def forward(self, x: torch.Tensor, length=None):
        """
        Forward pass of the MPNCOV module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying covariance pooling, optional square root normalization,
                          and optional vectorization.
        """
        if x.ndim == 3: # Audio (B, D, T)
            x = x.unsqueeze(2)
        if self.dimension_reduction is not None:
            x = self.conv_dim_reduction(x)
        x = self._covariance_pooling(x)
        if self.use_sqrt:
            x = self._matrix_square_root(x)
        if self.use_vector:
            x = self._upper_triangular_vector(x)
        return x


class CovPoolingFunction(Function):
    """
    Autograd Function for covariance pooling.
    
    Computes the covariance matrix of input features.
    """

    @staticmethod
    def forward(ctx, input_tensor):
        """
        Forward pass for covariance pooling.

        Args:
            input_tensor (torch.Tensor): Input feature map with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Covariance matrix with shape (batch_size, channels, channels).
        """
        batch_size, channels, height, width = input_tensor.shape
        num_elements = height * width

        # Reshape input to (batch_size, channels, num_elements)
        x_reshaped = input_tensor.view(batch_size, channels, num_elements)

        # Create the centering matrix:
        # I_hat = (-1/(M^2)) * ones(M, M) + (1/M) * eye(M, M)
        eye_matrix = torch.eye(num_elements, device=input_tensor.device, dtype=input_tensor.dtype)
        ones_matrix = torch.ones((num_elements, num_elements), device=input_tensor.device, dtype=input_tensor.dtype)
        I_hat = (-1.0 / (num_elements * num_elements)) * ones_matrix + (1.0 / num_elements) * eye_matrix
        I_hat = I_hat.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute the covariance: X * I_hat * X^T
        covariance = x_reshaped.bmm(I_hat).bmm(x_reshaped.transpose(1, 2))
        ctx.save_for_backward(input_tensor, I_hat)
        return covariance

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for covariance pooling.

        Args:
            grad_output (torch.Tensor): Gradient of the loss with respect to the output,
                                        shape (batch_size, channels, channels).

        Returns:
            torch.Tensor: Gradient of the loss with respect to the input,
                          shape (batch_size, channels, height, width).
        """
        input_tensor, I_hat = ctx.saved_tensors
        batch_size, channels, height, width = input_tensor.shape
        num_elements = height * width

        # Reshape input tensor to (batch_size, channels, num_elements)
        x_reshaped = input_tensor.view(batch_size, channels, num_elements)

        # Ensure symmetry of the gradient
        grad_sym = grad_output + grad_output.transpose(1, 2)
        grad_input = grad_sym.bmm(x_reshaped).bmm(I_hat)
        grad_input = grad_input.view(batch_size, channels, height, width)
        return grad_input


class MatrixSquareRootFunction(Function):
    """
    Autograd Function for computing the matrix square root via Newton-Schulz iteration.
    """

    @staticmethod
    def forward(ctx, input_tensor, iter_num):
        """
        Forward pass to compute the matrix square root.

        Args:
            input_tensor (torch.Tensor): Input covariance matrix with shape (batch_size, channels, channels).
            iter_num (int): Number of iterations for the Newton-Schulz method.

        Returns:
            torch.Tensor: Matrix square root of the input.
        """
        batch_size, channels, _ = input_tensor.shape
        dtype = input_tensor.dtype
        device = input_tensor.device

        # Compute 3*I (used for normalization)
        I3 = 3.0 * torch.eye(channels, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)

        # Normalize the input matrix: norm_A = (1/3) * trace(3*I * input)
        norm_A = (1.0 / 3.0) * (input_tensor * I3).sum(dim=1).sum(dim=1)
        norm_A = norm_A.view(batch_size, 1, 1)
        A = input_tensor / norm_A

        # Initialize Y and Z for the iterative process
        Y = torch.zeros(batch_size, iter_num, channels, channels, device=device, dtype=dtype)
        Z = torch.eye(channels, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).repeat(batch_size, iter_num, 1, 1)

        if iter_num < 2:
            # If fewer than 2 iterations, perform one step.
            ZY = 0.5 * (I3 - A)
            YZY = A.bmm(ZY)
        else:
            # First iteration
            ZY = 0.5 * (I3 - A)
            Y[:, 0, :, :] = A.bmm(ZY)
            Z[:, 0, :, :] = ZY

            # Iteratively update Y and Z
            for i in range(1, iter_num - 1):
                ZY = 0.5 * (I3 - Z[:, i - 1, :, :].bmm(Y[:, i - 1, :, :]))
                Y[:, i, :, :] = Y[:, i - 1, :, :].bmm(ZY)
                Z[:, i, :, :] = ZY.bmm(Z[:, i - 1, :, :])
            YZY = 0.5 * Y[:, iter_num - 2, :, :].bmm(I3 - Z[:, iter_num - 2, :, :].bmm(Y[:, iter_num - 2, :, :]))

        # Scale the result back with the square root of norm_A
        sqrt_norm_A = torch.sqrt(norm_A)
        output = YZY * sqrt_norm_A.expand_as(input_tensor)

        # Save tensors for backward pass
        ctx.save_for_backward(input_tensor, A, YZY, norm_A, Y, Z)
        ctx.iter_num = iter_num
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the matrix square root function.

        Args:
            grad_output (torch.Tensor): Gradient of the loss with respect to the output,
                                        shape (batch_size, channels, channels).

        Returns:
            Tuple[torch.Tensor, None]: Gradient with respect to the input tensor and None (for iter_num).
        """
        input_tensor, A, YZY, norm_A, Y, Z = ctx.saved_tensors
        iter_num = ctx.iter_num
        batch_size, channels, _ = input_tensor.shape
        dtype = input_tensor.dtype
        device = input_tensor.device

        # Scale the gradient by sqrt(norm_A)
        sqrt_norm_A = torch.sqrt(norm_A)
        der_post_com = grad_output * sqrt_norm_A.expand_as(input_tensor)

        # Compute an auxiliary gradient term per batch element
        der_post_com_aux = (grad_output * YZY).sum(dim=1).sum(dim=1) / (2 * sqrt_norm_A.view(batch_size))

        # Precompute constant matrix 3*I
        I3 = 3.0 * torch.eye(channels, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)

        if iter_num < 2:
            der_ns_iter = 0.5 * (der_post_com.bmm(I3 - A) - A.bmm(der_post_com))
        else:
            dLdY = 0.5 * (der_post_com.bmm(I3 - Y[:, iter_num - 2, :, :].bmm(Z[:, iter_num - 2, :, :])) -
                          Z[:, iter_num - 2, :, :].bmm(Y[:, iter_num - 2, :, :]).bmm(der_post_com))
            dLdZ = -0.5 * Y[:, iter_num - 2, :, :].bmm(der_post_com).bmm(Y[:, iter_num - 2, :, :])
            # Backpropagate through the Newton-Schulz iterations
            for i in range(iter_num - 3, -1, -1):
                YZ = I3 - Y[:, i, :, :].bmm(Z[:, i, :, :])
                ZY = Z[:, i, :, :].bmm(Y[:, i, :, :])
                dLdY_next = 0.5 * (dLdY.bmm(YZ) - Z[:, i, :, :].bmm(dLdZ).bmm(Z[:, i, :, :]) - ZY.bmm(dLdY))
                dLdZ_next = 0.5 * (YZ.bmm(dLdZ) - Y[:, i, :, :].bmm(dLdY).bmm(Y[:, i, :, :]) - dLdZ.bmm(ZY))
                dLdY, dLdZ = dLdY_next, dLdZ_next
            der_ns_iter = 0.5 * (dLdY.bmm(I3 - A) - dLdZ - A.bmm(dLdY))

        # Transpose and rescale gradient
        der_ns_iter = der_ns_iter.transpose(1, 2)
        grad_input = der_ns_iter / norm_A.expand_as(input_tensor)

        # Adjust gradient with the auxiliary term
        grad_aux = (der_ns_iter * input_tensor).sum(dim=1).sum(dim=1)
        for i in range(batch_size):
            grad_input[i, :, :] += (
                der_post_com_aux[i] - grad_aux[i] / (norm_A[i] * norm_A[i])
            ) * torch.eye(channels, device=device, dtype=dtype)
        return grad_input, None


class UpperTriangularVectorFunction(Function):
    """
    Autograd Function for extracting and vectorizing the upper triangular part of a matrix.
    """

    @staticmethod
    def forward(ctx, input_tensor):
        """
        Forward pass to extract the upper triangular (including diagonal) elements and vectorize them.

        Args:
            input_tensor (torch.Tensor): Input matrix of shape (batch_size, channels, channels).

        Returns:
            torch.Tensor: Vectorized upper triangular part with shape (batch_size, channels*(channels+1)//2).
        """
        batch_size, channels, _ = input_tensor.shape
        dtype = input_tensor.dtype

        # Flatten each matrix to shape (batch_size, channels*channels)
        x_flat = input_tensor.view(batch_size, channels * channels)

        # Obtain indices for the upper triangular part using torch.triu_indices
        row_idx, col_idx = torch.triu_indices(channels, channels, device=input_tensor.device)
        # Convert 2D indices to 1D linear indices
        indices = row_idx * channels + col_idx

        # Select the upper triangular elements
        output = x_flat[:, indices]
        ctx.save_for_backward(input_tensor, indices)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the upper triangular vectorization.

        Args:
            grad_output (torch.Tensor): Gradient of the loss with respect to the output,
                                        shape (batch_size, num_upper_elements).

        Returns:
            torch.Tensor: Gradient of the loss with respect to the input,
                          shape (batch_size, channels, channels).
        """
        input_tensor, indices = ctx.saved_tensors
        batch_size, channels, _ = input_tensor.shape
        dtype = input_tensor.dtype
        device = input_tensor.device

        # Initialize gradient for the flattened input with zeros
        grad_input_flat = torch.zeros(batch_size, channels * channels, device=device, dtype=dtype)
        grad_input_flat[:, indices] = grad_output
        # Reshape back to the original matrix shape
        grad_input = grad_input_flat.view(batch_size, channels, channels)
        return grad_input


# Utility layer wrappers for easier access to the functions

def cov_pool_layer(input_tensor):
    """
    Covariance pooling layer wrapper.

    Args:
        input_tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Covariance pooled tensor.
    """
    return CovPoolingFunction.apply(input_tensor)


def matrix_sqrt_layer(input_tensor, iter_num):
    """
    Matrix square root layer wrapper.

    Args:
        input_tensor (torch.Tensor): Input tensor.
        iter_num (int): Number of iterations for the Newton-Schulz method.

    Returns:
        torch.Tensor: Matrix square root of the input tensor.
    """
    return MatrixSquareRootFunction.apply(input_tensor, iter_num)


def upper_triangular_vector_layer(input_tensor):
    """
    Upper triangular vectorization layer wrapper.

    Args:
        input_tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Vectorized upper triangular part of the input tensor.
    """
    return UpperTriangularVectorFunction.apply(input_tensor)
