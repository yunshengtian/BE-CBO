import math
import torch
from torch import Tensor
from .base import ConstrainedSyntheticTestFunction


class LSQ(ConstrainedSyntheticTestFunction):
    """
    LSQ: linear objective, sinusoidal and quadratic inequality constraints
    [Gramacy et al. 2015] Modeling an Augmented Lagrangian for Blackbox Constrained Optimization
    """
    dim = 2
    _bounds = [(0, 1), (0, 1)]
    _optimal_value = 0.5998
    _optimizers = [(0.1954, 0.4044)]
    name = 'LSQ Function'

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = X[:, 0] + X[:, 1]
        return f

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        cons = torch.stack((
            X[:, 0] + 2 * X[:, 1] + 0.5 * torch.sin(2 * math.pi * (X[:, 0] ** 2 - 2 * X[:, 1])) - 1.5,
            1.5 - X[:, 0] ** 2 - X[:, 1] ** 2,
        ), dim=1)
        return cons


class LSQShift(LSQ):

    _optimal_value = 0.0
    _optimizers = [(0.4, 0.45)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        f = (X[:, 0] - 0.4) ** 2 + (X[:, 1] - 0.45) ** 2
        return f
