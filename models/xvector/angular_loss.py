import torch
import torch.nn as nn


class Loss(nn.modules.loss._Loss):
    """Inherit this class to implement custom loss."""

    def __init__(self, **kwargs):
        super(Loss, self).__init__(**kwargs)


class AngularSoftmaxLoss(Loss):
    """Additive Angular Margin Softmax (ArcFace).

    Paper: Deng, Jiankang, et al. "Arcface: Additive angular margin loss for deep face recognition." 
    Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.

    Args:
    scale: scale value for cosine angle
    margin: margin value added to cosine angle 
    """

    def __init__(self, scale: float = 20.0, margin: float = 1.35):
        super().__init__()
        self.eps = 1e-7
        self.scale = scale
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        numerator = self.scale * torch.cos(
            torch.acos(torch.clamp(torch.diagonal(logits.transpose(0, 1)[labels]), -1.0 + self.eps, 1 - self.eps))
            + self.margin
        )
        excl = torch.cat(
            [torch.cat((logits[i, :y], logits[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
        