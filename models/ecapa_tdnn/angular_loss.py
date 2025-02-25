import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_jacobi, eval_jacobi

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
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class Poly1CrossEntropyLoss(Loss):

    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # Apply polynomial loss function
        pt = torch.exp(-ce_loss)
        poly_loss = ce_loss + self.epsilon * (1 - pt)
        return torch.mean(poly_loss)


class Poly1FocalLoss(Loss):

    def __init__(self, alpha=0.25, gamma=2.0, epsilon=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # Compute focal loss
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        # Apply polynomial loss function
        poly_loss = focal_loss + self.epsilon * torch.pow(1 - pt, self.gamma + 1)
        return torch.mean(poly_loss)


class NormFaceLoss(Loss):
    """Computes NormFace Loss
    
    Paper: NormFace: L2 hypersphere embedding for face verification (MM'17)

    args:
    scale: scale value for cosine angle
    margin: angular margin multiplied with cosine angle
    """

    def __init__(self, scale=30.0):
        super().__init__()
        self.eps = 1e-7
        self.scale = scale

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # Extract the logits corresponding to the true class
        cos_theta_target = torch.diagonal(logits.transpose(0, 1)[labels])
        numerator = self.scale * cos_theta_target
        # Exclude the target logits from denominator calculation
        cos_theta_others = torch.cat(
            [torch.cat((logits[i, :y], logits[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * cos_theta_others), dim=1)
        # Compute final loss
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


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
        k = torch.floor(theta_target * self.margin / torch.pi)
        sign = torch.where(k % 2 == 0, 1, -1)
        numerator = self.scale * (sign * torch.cos(self.margin * theta_target) - 2 * k)
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


def bounded_chebyshev_approximation(func, degree=30, num_samples=1000, margin=0.2):
    x = np.linspace(-1, 1, num_samples)
    y = func(x, margin)
    cheb = np.polynomial.Chebyshev.fit(x, y, degree, domain=[-1, 1])
    return cheb.coef


class ChebyshevV1ArcFaceLoss(Loss):

    def __init__(self, scale=30.0, margin=0.2, chebyshev_degree=10):
        super().__init__()
        self.eps = 1e-7
        self.scale = scale
        self.margin = margin

        def original_func(u, margin):
            u_clipped = np.clip(u, -1, 1)
            return np.cos(np.arccos(u_clipped) + margin)
        
        # Precompute Chebyshev coefficients
        cheb_coeffs_np = bounded_chebyshev_approximation(
            original_func, degree=chebyshev_degree, margin=margin
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
        def original_func(cos_theta):
            cos_theta = np.clip(cos_theta, -1+self.eps, 1-self.eps)
            return cos_theta * np.cos(margin) - np.sqrt(1 - cos_theta**2) * np.sin(margin)
        
        # Compute optimized polynomial coefficients
        self.poly_coeffs = compute_remez_poly(original_func, interval, remez_degree)
        
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


class ChebyshevArcFaceLoss(Loss):

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

        def target_func(u, margin):
            """
            Computes the target function f(u) = cos(arccos(u) + margin),
            clipping u into [-1, 1] for numerical stability.
            """
            u_clipped = np.clip(u, -1, 1)
            return np.cos(np.arccos(u_clipped) + margin)
        
        # Precompute Chebyshev coefficients using Clenshaw–Curtis nodes
        cheb_coeffs_np = clenshaw_curtis_chebyshev_coefficients(
            target_func, degree=chebyshev_degree, num_samples=num_samples, margin=margin
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


def legendre_coefficients(func, degree=30, num_samples=1000, margin=0.2):
    """
    Compute Legendre series coefficients using Gauss-Legendre quadrature.
    
    Args:
        func: Target function to approximate.
        degree: Maximum degree of the Legendre polynomial series.
        num_samples: Number of quadrature points.
        margin: Margin parameter for the ArcFace loss.
    
    Returns:
        np.ndarray: Array of Legendre coefficients.
    """
    nodes, weights = np.polynomial.legendre.leggauss(num_samples)
    y = func(nodes, margin)
    coeffs = []
    for k in range(degree + 1):
        # Evaluate k-th Legendre polynomial at nodes
        P_k = np.polynomial.legendre.legval(nodes, [0]*k + [1])
        # Compute the integral via quadrature
        integral = np.sum(weights * y * P_k)
        a_k = (2*k + 1) / 2 * integral
        coeffs.append(a_k)
    return np.array(coeffs)


class LegendreDirectFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, coeffs):
        """
        Forward pass: Evaluate Legendre series at x.
        
        Args:
            x: Input tensor in [-1 + eps, 1 - eps].
            coeffs: Tensor of Legendre coefficients.
        
        Returns:
            Tensor: Evaluated series.
        """
        n = coeffs.shape[0] - 1
        P = [torch.ones_like(x)]  # P_0
        if n >= 1:
            P.append(x)  # P_1
        for k in range(2, n + 1):
            Pk = ((2*k - 1) * x * P[k-1] - (k - 1) * P[k-2]) / k
            P.append(Pk)
        # Compute the series sum
        f = sum(coeffs[k] * P[k] for k in range(n + 1))
        ctx.save_for_backward(x, coeffs)
        ctx.P = P
        ctx.n = n
        return f

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Compute gradient with respect to x.
        """
        x, coeffs = ctx.saved_tensors
        P, n = ctx.P, ctx.n
        derivative = torch.zeros_like(x)
        for k in range(1, n + 1):
            # Ensure denominator is non-zero (x is clipped in practice)
            denominator = 1 - x**2
            # Add small epsilon to avoid division by zero
            denominator = torch.where(denominator > 1e-6, denominator, torch.ones_like(denominator) * 1e-6)
            Pk_prime = (k * P[k-1] - k * x * P[k]) / denominator
            derivative += coeffs[k] * Pk_prime
        grad_input = grad_output * derivative
        return grad_input, None  # No gradient for coeffs


class LegendreArcFaceLoss(Loss):
    
    def __init__(self, scale=30.0, margin=0.2, legendre_degree=10, num_samples=1000):
        """
        ArcFace loss with Legendre polynomial approximation.
        
        Args:
            scale: Scaling factor for logits.
            margin: Margin parameter m.
            legendre_degree: Degree of the Legendre series.
            num_samples: Number of quadrature points for coefficient computation.
        """
        super().__init__()
        self.eps = 1e-7
        self.scale = scale
        self.margin = margin

        def target_func(u, margin):
            """
            Computes the target function f(u) = cos(arccos(u) + margin),
            clipping u into [-1, 1] for numerical stability.
            """
            u_clipped = np.clip(u, -1, 1)
            return np.cos(np.arccos(u_clipped) + margin)
        
        # Precompute coefficients
        leg_coeffs_np = legendre_coefficients(target_func, degree=legendre_degree, num_samples=num_samples, margin=margin)
        self.register_buffer('coefficients', torch.from_numpy(leg_coeffs_np).float())
    
    def legendre_eval(self, x):
        """Evaluate the Legendre series at x."""
        return LegendreDirectFunction.apply(x, self.coefficients)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Compute the ArcFace loss.
        
        Args:
            logits: Cosine similarities (batch_size, num_classes).
            labels: Ground truth labels (batch_size,).
        
        Returns:
            Tensor: Mean negative log likelihood loss.
        """
        # Extract target cosine similarities
        cos_theta_target = torch.diagonal(logits.transpose(0, 1)[labels])
        cos_theta_target = cos_theta_target.clamp(-1.0 + self.eps, 1 - self.eps)
        
        # Apply Legendre approximation and scaling
        numerator = self.scale * self.legendre_eval(cos_theta_target)
        
        # Compute denominator (excluding target class)
        cos_theta_others = torch.cat([
            torch.cat((logits[i, :y], logits[i, y+1:])).unsqueeze(0)
            for i, y in enumerate(labels)
        ], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * cos_theta_others), dim=1)
        
        # Loss computation
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


def jacobi_coefficients(func, alpha, beta, degree=30, num_samples=1000, margin=0.2):
    """
    Compute Jacobi series coefficients using Gauss-Jacobi quadrature.
    
    Args:
        func: Target function to approximate.
        alpha (float): First parameter of the Jacobi polynomials (alpha > -1).
        beta (float): Second parameter of the Jacobi polynomials (beta > -1).
        degree (int): Maximum degree of the Jacobi polynomial series.
        num_samples (int): Number of quadrature points.
        margin (float): Margin parameter for the ArcFace loss.
    
    Returns:
        np.ndarray: Array of Jacobi coefficients.
    """
    # Get quadrature nodes and weights
    nodes, weights = roots_jacobi(num_samples, alpha, beta)
    y = func(nodes, margin)
    coeffs = []
    
    # Compute coefficients for each degree
    for k in range(degree + 1):
        P_k = eval_jacobi(k, alpha, beta, nodes)
        integral_approx = np.sum(weights * y * P_k)
        # Compute the norm using quadrature for numerical stability
        h_k = np.sum(weights * P_k**2)
        a_k = integral_approx / h_k
        coeffs.append(a_k)
    return np.array(coeffs)


class JacobiDirectFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, coeffs, alpha, beta):
        """
        Forward pass: Evaluate Jacobi series at x.
        
        Args:
            x (torch.Tensor): Input tensor in [-1 + eps, 1 - eps].
            coeffs (torch.Tensor): Tensor of Jacobi coefficients.
            alpha (float): First parameter of the Jacobi polynomials.
            beta (float): Second parameter of the Jacobi polynomials.
        
        Returns:
            torch.Tensor: Evaluated series.
        """
        n = coeffs.shape[0] - 1
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype
        
        # Initialize polynomials and their derivatives
        P = torch.zeros(n + 1, batch_size, device=device, dtype=dtype)
        P_prime = torch.zeros(n + 1, batch_size, device=device, dtype=dtype)
        P[0] = 1.0
        P_prime[0] = 0.0
        
        if n >= 1:
            alpha_beta_sum = alpha + beta
            P[1] = ((alpha_beta_sum + 2)/2 * x + (alpha - beta)/2)
            P_prime[1] = (alpha_beta_sum + 2)/2
        
        # Compute higher-degree polynomials using recurrence
        for k in range(2, n + 1):
            temp = 2*k + alpha + beta - 1
            A = temp * ((2*k + alpha + beta) * (2*k + alpha + beta - 2) * x + (alpha**2 - beta**2))
            B = 2 * (k - 1) * (k + alpha - 1) * (k + beta - 1) * (2*k + alpha + beta)
            D = 2 * k * (k + alpha + beta) * (2*k + alpha + beta - 2)
            P[k] = (A * P[k-1] - B * P[k-2]) / D
            A_prime = temp * (2*k + alpha + beta) * (2*k + alpha + beta - 2)
            P_prime[k] = (A_prime * P[k-1] + A * P_prime[k-1] - B * P_prime[k-2]) / D
        
        # Evaluate the series
        f = torch.sum(coeffs[:, None] * P, dim=0)
        
        # Save for backward pass
        ctx.save_for_backward(P_prime, coeffs)
        ctx.n = n
        return f

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Compute gradient with respect to x.
        
        Args:
            grad_output (torch.Tensor): Gradient from the next layer.
        
        Returns:
            tuple: Gradients with respect to inputs (only x has gradient).
        """
        P_prime, coeffs = ctx.saved_tensors
        n = ctx.n
        derivative = torch.sum(coeffs[1:n+1, None] * P_prime[1:n+1], dim=0)
        grad_input = grad_output * derivative
        return grad_input, None, None, None  # No gradients for coeffs, alpha, beta


class JacobiArcFaceLoss(Loss):

    def __init__(self, scale=30.0, margin=0.2, alpha=0.0, beta=0.0, jacobi_degree=5, num_samples=1000):
        """
        ArcFace loss with Jacobi polynomial approximation.
        
        Args:
            scale (float): Scaling factor for logits.
            margin (float): Margin parameter m.
            alpha (float): First parameter of the Jacobi polynomials (alpha > -1).
            beta (float): Second parameter of the Jacobi polynomials (beta > -1).
            jacobi_degree (int): Degree of the Jacobi series.
            num_samples (int): Number of quadrature points for coefficient computation.
        """
        super().__init__()
        self.eps = 1e-7
        self.scale = scale
        self.margin = margin
        self.alpha = alpha
        self.beta = beta

        def target_func(u, margin):
            """
            Computes the target function f(u) = cos(arccos(u) + margin),
            clipping u into [-1, 1] for numerical stability.
            """
            u_clipped = np.clip(u, -1, 1)
            return np.cos(np.arccos(u_clipped) + margin)
        
        # Precompute Jacobi coefficients
        jacobi_coeffs_np = jacobi_coefficients(target_func, alpha, beta, degree=jacobi_degree, num_samples=num_samples, margin=margin)
        self.register_buffer('coefficients', torch.from_numpy(jacobi_coeffs_np).float())
    
    def jacobi_eval(self, x):
        """
        Evaluate the Jacobi series at x.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Evaluated series.
        """
        return JacobiDirectFunction.apply(x, self.coefficients, self.alpha, self.beta)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Compute the ArcFace loss.
        
        Args:
            logits (torch.Tensor): Cosine similarities of shape (batch_size, num_classes).
            labels (torch.Tensor): Ground truth labels of shape (batch_size,).
        
        Returns:
            torch.Tensor: Mean negative log likelihood loss.
        """
        # Extract target cosine similarities
        cos_theta_target = torch.diagonal(logits.transpose(0, 1)[labels])
        cos_theta_target = cos_theta_target.clamp(-1.0 + self.eps, 1 - self.eps)
        
        # Apply Jacobi approximation and scaling
        numerator = self.scale * self.jacobi_eval(cos_theta_target)
        
        # Compute denominator (excluding target class)
        cos_theta_others = torch.cat([
            torch.cat((logits[i, :y], logits[i, y+1:])).unsqueeze(0)
            for i, y in enumerate(labels)
        ], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * cos_theta_others), dim=1)
        
        # Compute loss
        L = numerator - torch.log(denominator)
        return -torch.mean(L)