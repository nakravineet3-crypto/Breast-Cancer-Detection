"""
Trainable Self-Guided Filter (TSGF) — image preprocessing layer.

Implements the paper's Eqs. 1-5 as a self-guided edge-preserving filter:
for every pixel, compute local linear coefficients (a, b) such that
output = a * input + b, where `a` near 1 keeps real edges sharp and
`a` near 0 smooths likely noise. The smoothing strength (`eps`) is a
learnable parameter, trained jointly with the rest of the network.

Note: the paper's equations for TSGF are loosely specified (several
symbols are reused/underdefined). This is a faithful, standard
self-guided-filter formulation matching the paper's stated *behavior*
(normalize, denoise, preserve structure) rather than a symbol-for-symbol
transcription — worth mentioning if your professor asks about exact
equation correspondence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainableSelfGuidedFilter(nn.Module):
    def __init__(self, radius: int = 4, eps_init: float = 0.01):
        super().__init__()
        self.radius = radius
        # store log(eps) and exponentiate later, so eps always stays positive
        # no matter what value gradient descent nudges the parameter to
        self.log_eps = nn.Parameter(torch.log(torch.tensor(float(eps_init))))

    def _box_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Local average over a (2*radius+1) x (2*radius+1) window, same size out."""
        k = 2 * self.radius + 1
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=self.radius, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, height, width), values in [0, 1]."""
        eps = torch.exp(self.log_eps)

        mean_I = self._box_filter(x)
        corr_I = self._box_filter(x * x)
        var_I = corr_I - mean_I * mean_I

        a = var_I / (var_I + eps)
        b = mean_I - a * mean_I

        mean_a = self._box_filter(a)
        mean_b = self._box_filter(b)

        q = mean_a * x + mean_b
        return torch.clamp(q, 0.0, 1.0)
