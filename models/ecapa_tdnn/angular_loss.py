import math

import mpmath
import numpy as np

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


def compute_pade_coefficients(func, order):
    """
    Computes Padé approximant coefficients for a given function.

    Args:
        func: The function to approximate (mpmath function).
        order: Tuple (m, n) for the Padé approximant (degree of numerator, denominator).

    Returns:
        Tuple of lists (p_coeffs, q_coeffs) for numerator and denominator polynomials.
    """
    mpmath.mp.dps = 50  # High precision for coefficient computation
    taylor_series = mpmath.taylor(func, 0, sum(order))  # Compute Maclaurin series
    p, q = mpmath.pade(taylor_series, order[0], order[1])  # Compute Padé approximant
    p, q = [float(c) for c in p], [float(c) for c in q]  # Convert to standard float
    return p, q


def horner_evaluation(x, coeffs):
    """
    Evaluates a polynomial using Horner's method.

    Args:
        x: The input tensor.
        coeffs: List of polynomial coefficients (constant term first).

    Returns:
        Evaluated polynomial value.
    """
    result = torch.zeros_like(x, dtype=x.dtype)
    for coef in reversed(coeffs): # Horner’s method should process from highest-degree term first to lowest.
        result = result * x + coef
    return result


class PadeCosine(torch.nn.Module):

    def __init__(self, order=(4, 4), dtype=torch.float32):
        """
        Constructs a Padé approximant model for cosine.

        Args:
            order: Tuple specifying degrees of numerator and denominator.
            dtype: Data type (float32 or float16).
        """
        super().__init__()
        p_coeffs, q_coeffs = compute_pade_coefficients(mpmath.cos, order)
        logger.info("Padé approximant for cosine: ")
        logger.info(f"P = {p_coeffs}")
        logger.info(f"Q = {q_coeffs}")

        # Convert coefficients to PyTorch tensors
        # self.p_coeffs = torch.tensor(p_coeffs, dtype=dtype)
        # self.q_coeffs = torch.tensor(q_coeffs, dtype=dtype)
        self.register_buffer('p_coeffs', torch.tensor(p_coeffs, dtype=dtype))
        self.register_buffer('q_coeffs', torch.tensor(q_coeffs, dtype=dtype))

    def forward(self, x):
        # Cosine is an even function, meaning it should use x^2 for stability
        x2 = x * x
        x2 = x2.clamp(-1 + 1e-6, 1 - 1e-6) # Fix: Prevent instability
        num = horner_evaluation(x, self.p_coeffs)
        denom = horner_evaluation(x, self.q_coeffs)
        return num / denom

class PadeArccos(torch.nn.Module):

    def __init__(self, order=(4, 4), dtype=torch.float32):
        """
        Constructs a Padé approximant model for arccosine.

        Args:
            order: Tuple specifying degrees of numerator and denominator.
            dtype: Data type (float32 or float16).
        """
        super().__init__()
        p_coeffs, q_coeffs = compute_pade_coefficients(mpmath.acos, order)
        logger.info("Padé approximant for arccosine: ")
        logger.info(f"P = {p_coeffs}")
        logger.info(f"Q = {q_coeffs}")

        # Convert coefficients to PyTorch tensors
        # self.p_coeffs = torch.tensor(p_coeffs, dtype=dtype)
        # self.q_coeffs = torch.tensor(q_coeffs, dtype=dtype)
        self.register_buffer('p_coeffs', torch.tensor(p_coeffs, dtype=dtype))
        self.register_buffer('q_coeffs', torch.tensor(q_coeffs, dtype=dtype))

    def forward(self, x):
        x = x.clamp(-1 + 1e-6, 1 - 1e-6)  # Fix: Prevent instability
        num = horner_evaluation(x, self.p_coeffs)
        denom = horner_evaluation(x, self.q_coeffs)
        return num / denom


class PadeArcFaceLoss(Loss):
    """Computes Additive Angular Margin Softmax (ArcFace) Loss using Padé approximants"""
    
    def __init__(self, scale=30.0, margin=0.2, fp16=False):
        super().__init__()
        self.eps = 1e-7
        self.scale = scale
        self.margin = margin

        dtype = torch.float16 if fp16 else torch.float32
        self.cos_approx = PadeCosine(order=(4, 4), dtype=dtype)
        self.acos_approx = PadeArccos(order=(4, 4), dtype=dtype)
        logger.info(f"PadeCosine and PadeArccos use dtype = {dtype}")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # Extract the logits corresponding to the true class
        cos_theta_target = logits[torch.arange(logits.size(0)), labels]
        numerator = self.scale * self.cos_approx(self.acos_approx(cos_theta_target) + self.margin)
        # Exclude the target logits from denominator calculation
        cos_theta_others = torch.cat(
            [torch.cat((logits[i, :y], logits[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0
        ) # without masking
        max_logit = self.scale * cos_theta_others
        denominator = torch.exp(numerator) + torch.sum(torch.exp(max_logit), dim=1)
        # Compute final loss
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


def taylor_cos(x, n_terms=10):
    """
    Compute the Taylor series expansion of cos(x) up to n_terms.
    """
    result = torch.zeros_like(x)
    for k in range(n_terms):
        term = ((-1) ** k) * (x ** (2 * k)) / math.factorial(2 * k)
        result += term
    return result


def taylor_arccos(x: torch.Tensor, n_terms=10):
    """
    Compute the Taylor series expansion of arccos(x) around x=0.
    """
    x = x.clamp(-1 + 1e-6, 1 - 1e-6)
    
    result = math.pi / 2 * torch.ones_like(x)
    for k in range(n_terms):
        coef = math.factorial(2 * k) / (2 ** (2 * k) * (math.factorial(k) ** 2) * (2 * k + 1))
        term = coef * (x ** (2 * k + 1))
        result -= term
    return result


class TaylorArcFaceLoss(Loss):
    """Computes Additive Angular Margin Softmax (ArcFace) Loss using Taylor series expansion"""
    
    def __init__(self, scale=30.0, margin=0.2, n_terms=10):
        super().__init__()
        self.eps = 1e-7
        self.scale = scale
        self.margin = margin
        self.n_terms = n_terms

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # Extract the logits corresponding to the true class
        cos_theta_target = logits[torch.arange(logits.size(0)), labels]
        numerator = self.scale * taylor_cos(taylor_arccos(cos_theta_target, self.n_terms) + self.margin, self.n_terms)
        # Exclude the target logits from denominator calculation
        cos_theta_others = torch.cat(
            [torch.cat((logits[i, :y], logits[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0
        ) # without masking
        max_logit = self.scale * cos_theta_others
        denominator = torch.exp(numerator) + torch.sum(torch.exp(max_logit), dim=1)
        # Compute final loss
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
    

def chebyshev_polynomials(x: torch.Tensor, n_terms):
    """
    Compute Chebyshev polynomials T_k(x) up to n_terms using recursion.
    """
    T = [torch.ones_like(x), x]
    for k in range(2, n_terms):
        T_next = 2 * x * T[-1] - T[-2]
        T.append(T_next)
    return T


def chebyshev_cos(x: torch.Tensor, n_terms=10):
    """
    Compute the Chebyshev series expansion of cos(x) up to n_terms.
    """
    T = chebyshev_polynomials(x, n_terms)
    coeffs = [math.cos(k * math.pi / 2) for k in range(n_terms)]
    result = sum(c * T_k for c, T_k in zip(coeffs, T))
    return result


def chebyshev_arccos(x: torch.Tensor, n_terms=10):
    """
    Compute the Chebyshev series expansion of arccos(x) up to n_terms.
    """
    x = x.clamp(-1 + 1e-6, 1 - 1e-6)
    
    T = chebyshev_polynomials(x, n_terms)
    coeffs = [math.pi / 2 if k == 0 else (-1)**k * 2 / (k * math.pi) for k in range(1, n_terms)]
    result = sum(c * T_k for c, T_k in zip([math.pi / 2] + coeffs, T))
    return result


class ChebyshevArcFaceLoss(Loss):
    """Computes Additive Angular Margin Softmax (ArcFace) Loss using Chebyshev series expansion"""
    
    def __init__(self, scale=30.0, margin=0.2, n_terms=10):
        super().__init__()
        self.eps = 1e-7
        self.scale = scale
        self.margin = margin
        self.n_terms = n_terms

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # Extract the logits corresponding to the true class
        cos_theta_target = logits[torch.arange(logits.size(0)), labels]
        numerator = self.scale * chebyshev_cos(chebyshev_arccos(cos_theta_target, self.n_terms) + self.margin, self.n_terms)
        # Exclude the target logits from denominator calculation
        cos_theta_others = torch.cat(
            [torch.cat((logits[i, :y], logits[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0
        ) # without masking
        max_logit = self.scale * cos_theta_others
        denominator = torch.exp(numerator) + torch.sum(torch.exp(max_logit), dim=1)
        # Compute final loss
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
    

def bhaskara_cos(x: torch.Tensor):
    """
    Compute Bhaskara's approximation for cos(x).
    """
    x = torch.remainder(x, 2 * math.pi)  # Keep x within [0, 2pi]
    numerator = 16 * (math.pi - x) * x
    denominator = 5 * math.pi**2 - 4 * x * (math.pi - x)
    return numerator / denominator


def bhaskara_arccos(x: torch.Tensor):
    """
    Approximate arccos(x) using Bhaskara's method.
    """
    x = x.clamp(-1 + 1e-6, 1 - 1e-6)
    return math.pi / 2 - (x + 0.25 * x**3)


class BhaskaraArcFaceLoss(Loss):
    """Computes Additive Angular Margin Softmax (ArcFace) Loss using Bhaskara series expansion"""
    
    def __init__(self, scale=30.0, margin=0.2):
        super().__init__()
        self.eps = 1e-7
        self.scale = scale
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # Extract the logits corresponding to the true class
        cos_theta_target = logits[torch.arange(logits.size(0)), labels]
        numerator = self.scale * bhaskara_cos(bhaskara_arccos(cos_theta_target) + self.margin)
        # Exclude the target logits from denominator calculation
        cos_theta_others = torch.cat(
            [torch.cat((logits[i, :y], logits[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0
        ) # without masking
        max_logit = self.scale * cos_theta_others
        denominator = torch.exp(numerator) + torch.sum(torch.exp(max_logit), dim=1)
        # Compute final loss
        L = numerator - torch.log(denominator)
        return -torch.mean(L)