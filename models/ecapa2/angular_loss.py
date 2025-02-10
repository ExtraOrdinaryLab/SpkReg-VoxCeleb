import torch
import torch.nn as nn


class Loss(nn.modules.loss._Loss):
    """Inherit this class to implement custom loss."""

    def __init__(self, **kwargs):
        super(Loss, self).__init__(**kwargs)


class AdditiveMarginSoftmaxLoss(Loss):
    """Computes Additive Margin Softmax (CosFace) Loss
    
    Paper: CosFace: Large Margin Cosine Loss for Deep Face Recognition

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
        logits_target = logits[torch.arange(logits.size(0)), labels]  # Faster indexing
        numerator = self.scale * (logits_target - self.margin)  # Apply additive margin
        # Exclude the target logits from denominator calculation
        logits.scatter_(1, labels.unsqueeze(1), float('-inf'))  # Mask target class
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * logits), dim=1)
        # Compute final loss
        loss = -torch.log(torch.exp(numerator) / denominator)
        return loss.mean()


class AdditiveAngularMarginSoftmaxLoss(Loss):
    """Computes Additive Angular Margin Softmax (ArcFace) Loss
    
    Paper: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    
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