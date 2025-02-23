import math

import mpmath
import numpy as np
import sympy as sp
from scipy.special import comb
from scipy.optimize import minimize
from scipy.interpolate import pade as scipy_pade

import torch
import torch.nn as nn
import torch.nn.functional as F

from .logging import logger


class Loss(nn.modules.loss._Loss):
    """Inherit this class to implement custom loss."""

    def __init__(self, **kwargs):
        super(Loss, self).__init__(**kwargs)


class FocalLoss(Loss):

    def __init__(self, alpha=1., gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return F_loss.mean()


class SphereFaceLoss(Loss):
    """Computes Multiplicative Angular Margin Softmax (SphereFace) Loss
    
    Paper: SphereFace: Deep Hypersphere Embedding for Face Recognition (CVPR'17)

    args:
    scale: scale value for cosine angle
    margin: angular margin multiplied with cosine angle
    """

    def __init__(self, scale=30.0, margin=1.35):
        super().__init__()
        self.eps = 1e-7
        self.scale = scale
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # Extract the logits corresponding to the true class
        cos_theta_target = torch.diagonal(logits.transpose(0, 1)[labels])
        theta_target = torch.acos(cos_theta_target.clamp(-1.0 + self.eps, 1.0 - self.eps))
        numerator = self.scale * torch.cos(self.margin * theta_target)
        # Exclude the target logits from denominator calculation
        cos_theta_others = torch.cat(
            [torch.cat((logits[i, :y], logits[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * cos_theta_others), dim=1)
        # Compute final loss
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class CosFaceLoss(Loss):
    """Computes Additive Margin Softmax (CosFace) Loss
    
    Paper: CosFace: Large Margin Cosine Loss for Deep Face Recognition (CVPR'18)

    args:
    scale: scale value for cosine angle
    margin: margin value added to cosine angle 
    """

    def __init__(self, scale=30.0, margin=0.2):
        super().__init__()
        self.eps = 1e-7
        self.scale = scale
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # Extract the logits corresponding to the true class
        cos_theta_target = torch.diagonal(logits.transpose(0, 1)[labels])
        numerator = self.scale * (cos_theta_target - self.margin)
        # Exclude the target logits from denominator calculation
        cos_theta_others = torch.cat(
            [torch.cat((logits[i, :y], logits[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * cos_theta_others), dim=1)
        # Compute final loss
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class NeMoArcFaceLoss(Loss):
    """Computes Additive Angular Margin Softmax (ArcFace) Loss
    
    Paper: ArcFace: Additive Angular Margin Loss for Deep Face Recognition (CVPR'19)
    
    Args:
    scale: scale value for cosine angle
    margin: margin value added to cosine angle 
    """

    def __init__(self, scale=20.0, margin=1.35):
        super().__init__()
        self.eps = 1e-7
        self.scale = scale
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # Extract the logits corresponding to the true class
        cos_theta_target = torch.diagonal(logits.transpose(0, 1)[labels])
        theta_target = torch.acos(cos_theta_target.clamp(-1.0 + self.eps, 1 - self.eps))
        numerator = self.scale * torch.cos(theta_target + self.margin)
        # Exclude the target logits from denominator calculation
        cos_theta_others = torch.cat(
            [torch.cat((logits[i, :y], logits[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * cos_theta_others), dim=1)
        # Compute final loss
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class SpeechBrainArcFaceLoss(Loss):
    """Computes Additive Angular Margin Softmax (ArcFace) Loss
    
    Paper: ArcFace: Additive Angular Margin Loss for Deep Face Recognition (CVPR'19)
    
    Args:
    scale: scale value for cosine angle
    margin: margin value added to cosine angle 
    """

    def __init__(self, scale=30.0, margin=0.2, dtype=torch.float32):
        super().__init__()
        self.eps = 1e-7
        self.scale = scale
        self.margin = margin

        self.register_buffer('cos_m', torch.tensor(math.cos(self.margin), dtype=dtype))
        self.register_buffer('sin_m', torch.tensor(math.sin(self.margin), dtype=dtype))
        self.register_buffer('threshold', torch.tensor(math.cos(math.pi - self.margin), dtype=dtype))
        self.register_buffer('m_margin', torch.tensor(math.sin(math.pi - self.margin) * self.margin, dtype=dtype))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # Extract the logits corresponding to the true class
        cos_theta = logits.float()
        cos_theta_target = torch.diagonal(cos_theta.transpose(0, 1)[labels])
        cos_theta_target = cos_theta_target.clamp(-1 + self.eps, 1 - self.eps)
        sin_theta_target = torch.sqrt(1.0 - torch.pow(cos_theta_target, 2))
        phi_target = cos_theta_target * self.cos_m - sin_theta_target * self.sin_m
        # Apply a decision rule for stability
        phi_target = torch.where(cos_theta_target > self.threshold, phi_target, cos_theta_target - self.m_margin)
        numerator = self.scale * phi_target
        # Exclude the target logits from denominator calculation
        cos_theta_others = torch.cat(
            [torch.cat((logits[i, :y], logits[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * cos_theta_others), dim=1)
        # Compute final loss
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class ArcFaceLoss(nn.Module):
    """Computes Additive Angular Margin Softmax (ArcFace) Loss
    
    Paper: ArcFace: Additive Angular Margin Loss for Deep Face Recognition (CVPR'19)

    Args:
    scale: scale value for cosine angle
    margin: angular margin added to cosine angle 
    """

    def __init__(self, scale=30.0, margin=0.5):
        super().__init__()
        self.eps = 1e-7
        self.scale = scale
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # Extract the logits corresponding to the true class
        # Convert sensitive calculations to float32 before applying fp16
        cos_theta_target = torch.diagonal(logits.transpose(0, 1)[labels])
        # cos_theta_target = cos_theta_target.clamp(-1.0 + self.eps, 1 - self.eps)
        theta_target = torch.acos(cos_theta_target)
        numerator = (self.scale * torch.cos(theta_target + self.margin))
        # Exclude the target logits from denominator calculation
        cos_theta_others = torch.cat(
            [torch.cat((logits[i, :y], logits[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * cos_theta_others), dim=1)
        # Compute final loss
        L = numerator - torch.log(denominator)
        # convert back to fp16 at the end:
        loss = -torch.mean(L)
        return loss


class LinearArcFaceLoss(Loss):
    """Computes Linear ArcFace Loss
    
    Paper: Airface: Lightweight and efficient model for face recognition (ICCV'19 Workshop)
    
    Args:
    scale: scale value for cosine angle
    margin: margin value added to cosine angle 
    """

    def __init__(self, scale=64.0, margin=0.4):
        super().__init__()
        self.eps = 1e-7
        self.scale = scale
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # Extract the logits corresponding to the true class
        cos_theta_target = torch.diagonal(logits.transpose(0, 1)[labels])
        # cos_theta_target = cos_theta_target.clamp(-1.0 + self.eps, 1 - self.eps)
        theta_target = torch.acos(cos_theta_target)
        numerator = self.scale * (torch.pi - 2 * (theta_target + self.margin)) / torch.pi
        # Exclude the target logits from denominator calculation
        cos_theta_others = torch.cat(
            [torch.cat((logits[i, :y], logits[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0
        )
        # cos_theta_others = cos_theta_others.clamp(-1.0 + self.eps, 1 - self.eps)
        logits_others = (torch.pi - 2 * torch.acos(cos_theta_others)) / torch.pi
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * logits_others), dim=1)
        # Compute final loss
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class AdaCosLoss(Loss):
    """Computes Adaptive Margin Softmax (AdaCos) Loss
    
    Paper: AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations (CVPR'19)

    args:
    initial_scale: initial scale value for cosine angle
    """

    def __init__(self, initial_scale=30.0):
        super().__init__()
        self.eps = 1e-7
        self.s = initial_scale

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # Compute theta (arccos of normalized dot product)
        theta = torch.acos(torch.clamp(logits, -1.0 + self.eps, 1.0 - self.eps))
        
        # Extract the logits corresponding to the true class
        theta_target = theta[torch.arange(logits.size(0)), labels]
        
        # Compute the adaptive scaling factor
        with torch.no_grad():
            one_hot = torch.zeros_like(logits)
            one_hot.scatter(1, labels.unsqueeze(1), 1)
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / logits.size(0)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        
        # Compute modified logits
        numerator = self.s * theta_target
        theta.scatter_(1, labels.unsqueeze(1), float('-inf'))  # Mask target class
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * theta), dim=1)
        
        # Compute final loss
        loss = -torch.log(torch.exp(numerator) / denominator)
        return loss.mean()


class QamFace(Loss):
    """Computes Quadratic Additive Angular Margin Softmax (Qamface) Loss

    Paper: Qamface: Quadratic additive angular margin loss for face recognition (ICIP'20)
    
    Args:
    scale: scale value for cosine angle
    margin: angular margin multiplied with cosine angle
    """
    def __init__(self, scale=6.0, margin=0.5):
        super().__init__()
        self.eps = 1e-7
        self.scale = scale
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # Extract the logits corresponding to the true class
        theta = torch.acos(torch.clamp(logits, -1.0 + self.eps, 1.0 - self.eps))
        theta_target = theta[torch.arange(theta.size(0)), labels]

        logits_target = (2 * torch.pi - (theta_target + self.margin)) ** 2  # Apply angular margin
        numerator = self.scale * logits_target  # Scale modified logits
        
        # Exclude the target logits from denominator calculation
        theta.scatter_(1, labels.unsqueeze(1), float('-inf'))  # Mask target class
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * (2 * torch.pi - theta) ** 2), dim=1)
        
        # Compute final loss
        loss = -torch.log(torch.exp(numerator) / denominator)
        return loss.mean()


def safe_original_func(u, margin):
    u_clipped = np.clip(u, -1, 1)
    return np.cos(np.arccos(u_clipped) + margin)


def bounded_chebyshev_approximation(func, degree=30, num_samples=1000, margin=0.2):
    x = np.linspace(-1, 1, num_samples)
    y = func(x, margin)
    cheb = np.polynomial.Chebyshev.fit(x, y, degree, domain=[-1, 1])
    return cheb.coef


class ChebyshevArcFaceLoss(Loss):

    def __init__(self, scale=30.0, margin=0.2, chebyshev_degree=10):
        super().__init__()
        self.eps = 1e-7
        self.scale = scale
        self.margin = margin
        
        # Precompute Chebyshev coefficients
        cheb_coeffs_np = bounded_chebyshev_approximation(
            safe_original_func, degree=chebyshev_degree, margin=margin
        )
        self.register_buffer('coefficients', torch.from_numpy(cheb_coeffs_np).float())
        
    def chebyshev_approx(self, x):
        coeffs = self.coefficients
        if len(coeffs) == 0:
            return torch.zeros_like(x)
        elif len(coeffs) == 1:
            return coeffs[0] * torch.ones_like(x)
        elif len(coeffs) == 2:
            return coeffs[0] + coeffs[1] * x
        
        x2 = 2 * x
        c0 = coeffs[-2].expand_as(x)
        c1 = coeffs[-1].expand_as(x)
        
        for i in range(3, len(coeffs) + 1):
            tmp = c0
            c0 = coeffs[-i] - c1
            c1 = tmp + c1 * x2
        
        return c0 + c1 * x

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # Extract target logits and ensure numerical stability
        cos_theta_target = torch.diagonal(logits.transpose(0, 1)[labels])
        cos_theta_target = cos_theta_target.clamp(-1.0 + self.eps, 1 - self.eps)
        
        # Compute numerator using Chebyshev approximation
        numerator = self.scale * self.chebyshev_approx(cos_theta_target)
        
        # Exclude target logits from denominator
        cos_theta_others = torch.cat([
            torch.cat((logits[i, :y], logits[i, y+1:])).unsqueeze(0) 
            for i, y in enumerate(labels)
        ], dim=0)
        
        # Compute denominator and loss
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * cos_theta_others), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


def compute_remez_poly(func, interval, degree, num_points=1000):
    """
    Compute minimax polynomial approximation using numerical optimization.
    Implements a simplified Remez-like algorithm using SciPy's minimize.
    """
    a, b = interval
    x = np.linspace(a, b, num_points)
    y = func(x)
    
    # Initial guess using Chebyshev nodes for better stability
    cheb_nodes = np.cos(np.pi * (2 * np.arange(1, degree+2) - 1) / (2 * (degree+1)))
    x_cheb = 0.5 * (b - a) * cheb_nodes + 0.5 * (a + b)
    coeffs_init = np.polyfit(x_cheb, func(x_cheb), degree)
    
    # Minimax optimization setup
    def objective(coeffs):
        approx = np.polyval(coeffs, x)
        errors = np.abs(approx - y)
        return np.max(errors) + 0.1*np.mean(errors)  # Combine max and mean error
    
    # Constrained optimization to maintain numerical stability
    result = minimize(
        objective,
        coeffs_init,
        method='SLSQP',
        options={'maxiter': 1000, 'ftol': 1e-12},
        bounds=[(None, None)]*len(coeffs_init)
    )
    
    if not result.success:
        print("Optimization warning:", result.message)
    
    return result.x


class RemezArcFaceLoss(nn.Module):
    """
    Improved ArcFace loss with minimax polynomial approximation using SciPy optimization.
    Maintains numerical stability and proper bounding.
    """
    def __init__(self, scale=30.0, margin=0.2, remez_degree=6, interval=(-1.0, 1.0)):
        super(RemezArcFaceLoss, self).__init__()
        self.eps = 1e-7
        self.scale = scale
        self.margin = margin
        self.interval = interval
        self.remez_degree = remez_degree

        # Define the target function with numerical safety
        def f(cos_theta):
            cos_theta = np.clip(cos_theta, -1+self.eps, 1-self.eps)
            return cos_theta * np.cos(margin) - np.sqrt(1 - cos_theta**2) * np.sin(margin)
        
        # Compute optimized polynomial coefficients
        self.poly_coeffs = compute_remez_poly(f, interval, remez_degree)
        
        # Register as buffer for proper device placement
        self.register_buffer('coefficients', torch.tensor(self.poly_coeffs, dtype=torch.float32))

    def remez_approx(self, x):

        def _eval_poly(coeffs, x):
            """
            Evaluate a polynomial at x given coefficients in descending order.
            For a polynomial P(x) = a_0*x^n + a_1*x^(n-1) + ... + a_n.
            """
            result = torch.zeros_like(x)
            degree = len(coeffs) - 1
            for i, coeff in enumerate(coeffs):
                result = result + coeff * (x ** (degree - i))
            return result
        
        x_clamped = x.clamp(self.interval[0], self.interval[1])
        return _eval_poly(self.coefficients, x_clamped)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # Extract target cosine similarities with numerical safety
        cos_theta_target = torch.diagonal(logits.transpose(0, 1))[labels]
        cos_theta_target = cos_theta_target.clamp(-1.0 + self.eps, 1 - self.eps)
        
        # Apply polynomial approximation
        approx_target = self.remez_approx(cos_theta_target)
        
        # Compute numerator and denominator with stability
        numerator = self.scale * approx_target
        
        # Exclude the target logits from denominator calculation
        cos_theta_others = torch.cat(
            [torch.cat((logits[i, :y], logits[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * cos_theta_others), dim=1)
        # Compute final loss
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


def safe_target_func(u, margin):
    """
    Computes the target function f(u) = cos(arccos(u) + margin),
    clipping u into [-1, 1] for numerical stability.
    """
    u_clipped = np.clip(u, -1, 1)
    return np.cos(np.arccos(u_clipped) + margin)


def clenshaw_curtis_chebyshev_coefficients(func, degree=30, num_samples=1000, margin=0.2):
    """
    Computes Chebyshev coefficients using Clenshaw-Curtis nodes.
    The nodes are given by: x_j = cos(pi * j / (num_samples-1)) for j = 0,...,num_samples-1.
    """
    j = np.arange(num_samples)
    x = np.cos(np.pi * j / (num_samples - 1))
    y = func(x, margin)
    coeffs = np.polynomial.chebyshev.chebfit(x, y, degree)
    return coeffs


class ChebyshevDirectFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, coeffs):
        """
        Directly evaluates the Chebyshev series
            f(x) = sum_{k=0}^{n} coeffs[k] * T_k(x),
        by computing T_k(x) via the recurrence:
            T_0(x)=1, T_1(x)=x, T_k(x)=2x*T_{k-1}(x)-T_{k-2}(x) for k>=2.
        """
        n = coeffs.shape[0] - 1
        T = []
        T0 = torch.ones_like(x)
        T.append(T0)
        if n >= 1:
            T1 = x
            T.append(T1)
        for k in range(2, n + 1):
            T_k = 2 * x * T[k - 1] - T[k - 2]
            T.append(T_k)
        # Compute f(x)
        f = sum(coeffs[k] * T[k] for k in range(n + 1))
        ctx.save_for_backward(x, coeffs)
        ctx.n = n
        return f

    @staticmethod
    def backward(ctx, grad_output):
        x, coeffs = ctx.saved_tensors
        n = ctx.n
        # Compute derivative: f'(x) = sum_{k=1}^{n} coeffs[k] * k * U_{k-1}(x)
        # U_0(x)=1, U_1(x)=2x, U_k(x)=2x*U_{k-1}(x)-U_{k-2}(x) for k>=2.
        U = []
        U0 = torch.ones_like(x)
        U.append(U0)
        if n >= 1:
            U1 = 2 * x
            U.append(U1)
        for k in range(2, n):
            U_k = 2 * x * U[k - 1] - U[k - 2]
            U.append(U_k)
        derivative = torch.zeros_like(x)
        for k in range(1, n + 1):
            derivative = derivative + coeffs[k] * k * U[k - 1]
        grad_input = grad_output * derivative
        return grad_input, None


class ChebyshevClenshawFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, coeffs):
        """
        Evaluates the Chebyshev series using Clenshaw's recurrence:
            b_k = coeffs[k] + 2*x*b_{k+1} - b_{k+2}, and
            f(x) = b_0 - x * b_2,
        iterating from k = n down to 0.
        """
        n = coeffs.shape[0] - 1
        b_kplus1 = torch.zeros_like(x)
        b_kplus2 = torch.zeros_like(x)
        x2 = 2 * x
        for k in range(n, -1, -1):
            b_k = coeffs[k] + x2 * b_kplus1 - b_kplus2
            b_kplus2 = b_kplus1
            b_kplus1 = b_k
        result = b_k - b_kplus2 * x
        ctx.save_for_backward(x, coeffs)
        ctx.n = n
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, coeffs = ctx.saved_tensors
        n = ctx.n
        # Use the same derivative expression:
        # f'(x) = sum_{k=1}^{n} coeffs[k] * k * U_{k-1}(x)
        U = []
        U0 = torch.ones_like(x)
        U.append(U0)
        if n >= 1:
            U1 = 2 * x
            U.append(U1)
        for k in range(2, n):
            U_k = 2 * x * U[k - 1] - U[k - 2]
            U.append(U_k)
        derivative = torch.zeros_like(x)
        for k in range(1, n + 1):
            derivative = derivative + coeffs[k] * k * U[k - 1]
        grad_input = grad_output * derivative
        return grad_input, None


class ChebyshevV2ArcFaceLoss(Loss):

    def __init__(self, scale=30.0, margin=0.2, chebyshev_degree=30, num_samples=1000, use_clenshaw=True):
        """
        Initializes ArcFaceLoss.
        Precomputes the Chebyshev coefficients to approximate:
            f(u) = cos(arccos(u) + margin)
        using Clenshaw-Curtis quadrature. Set use_clenshaw to False
        to use the direct evaluation method.
        """
        super().__init__()
        self.eps = 1e-7
        self.scale = scale
        self.margin = margin
        self.use_clenshaw = use_clenshaw
        
        # Precompute Chebyshev coefficients using Clenshaw–Curtis nodes
        cheb_coeffs_np = clenshaw_curtis_chebyshev_coefficients(
            safe_target_func, degree=chebyshev_degree, num_samples=num_samples, margin=margin
        )
        self.register_buffer('coefficients', torch.from_numpy(cheb_coeffs_np).float())
        
    def chebyshev_eval(self, x):
        """
        Evaluates the Chebyshev approximation using either the direct method
        or Clenshaw recurrence, along with custom autograd for the derivative.
        """
        if self.use_clenshaw:
            return ChebyshevClenshawFunction.apply(x, self.coefficients)
        else:
            return ChebyshevDirectFunction.apply(x, self.coefficients)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Computes the ArcFace loss using the Chebyshev approximation.
        For each sample, the target logit (cosine similarity) is approximated
        via the Chebyshev function evaluation, scaled, and then compared against
        the non-target logits.
        """
        # Extract target logits and ensure numerical stability
        cos_theta_target = torch.diagonal(logits.transpose(0, 1)[labels])
        cos_theta_target = cos_theta_target.clamp(-1.0 + self.eps, 1 - self.eps)
        
        # Compute the numerator using the Chebyshev approximation (with custom backward)
        numerator = self.scale * self.chebyshev_eval(cos_theta_target)
        
        # Exclude target logits from the denominator
        cos_theta_others = torch.cat([
            torch.cat((logits[i, :y], logits[i, y+1:])).unsqueeze(0)
            for i, y in enumerate(labels)
        ], dim=0)
        
        # Compute denominator and final loss
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * cos_theta_others), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)